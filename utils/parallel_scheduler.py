from __future__ import annotations

import io
import json
import logging
import signal
import subprocess
import time
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

from utils.configs import ParallelConfig
from utils.parallel_index import (
    day_folder,
    day_completed,
    load_index,
    index_path,
    atomic_write_json,
    mark_queued,
    mark_running,
    mark_done,
    mark_failed,
)


# =========
# Date utils
# =========

def expand_dates(d0: date, d1: date) -> List[date]:
    """Inclusive [d0, d1] date list."""
    out: List[date] = []
    cur = d0
    while cur <= d1:
        out.append(cur)
        cur = cur + timedelta(days=1)
    return out


def collect_eligible_days(cfg: ParallelConfig, log: logging.Logger) -> List[date]:
    """List of days we should schedule, subject to resume/limit_days."""
    all_days = expand_dates(cfg.start_date, cfg.end_date)
    eligible: List[date] = []
    for d in all_days:
        if cfg.resume and day_completed(cfg, d):
            log.info("Skip (already completed): %s", d.isoformat())
            continue
        eligible.append(d)
    if cfg.limit_days and len(eligible) > cfg.limit_days:
        log.info("Limiting days: taking first %d of %d eligible",
                 cfg.limit_days, len(eligible))
        eligible = eligible[:cfg.limit_days]
    return eligible


def echo_run_header(cfg: ParallelConfig, log: logging.Logger) -> None:
    payload = {
        "range": {"start_date": str(cfg.start_date), "end_date": str(cfg.end_date)},
        "concurrency": cfg.concurrency,
        "increment_size": cfg.increment_size,
        "resume": cfg.resume,
        "max_retries": cfg.max_retries,
        "stagger_seconds": cfg.stagger_seconds,
        "mem_guard_mb": cfg.mem_guard_mb,
        "dry_run": cfg.dry_run,
        "channels_file": str(cfg.channels_file),
        "ffl_spec": cfg.ffl_spec,
        "ffl_path": str(cfg.ffl_path),
        "out_dir": str(cfg.out_dir),
        "worker_path": str(cfg.worker_path),
        "run_utc_started": cfg.run_utc_started,
        "argv": cfg.args_echo,
        "limit_days": cfg.limit_days,
        "minutes_per_day": cfg.minutes_per_day,
        "compression": cfg.compression,
        "add_pruned": cfg.add_pruned,
    }
    log.info("===== parallel conversion: start =====")
    log.info(json.dumps(payload, indent=2))


# =========
# Memory guard
# =========

def mem_available_mb() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    mem_value = int(parts[1]) // 1024  # kB -> MB
                    logging.getLogger("parallel").debug(
                        "Memory available: %d MB", mem_value
                    )
                    return mem_value
    except Exception:
        return 1_000_000  # if unknown, pretend plenty
    return 0


def mem_ok(threshold_mb: int) -> bool:
    if threshold_mb <= 0:
        return True
    available_mb = mem_available_mb()
    log = logging.getLogger("parallel")
    log.info("Memory available: %d MB (guard threshold: %d MB)",
             available_mb, threshold_mb)
    return available_mb >= threshold_mb


# =========
# Worker command
# =========

def build_worker_cmd(cfg: ParallelConfig, d: date) -> list[str]:
    """
    Build the CLI command for the single-day worker, matching the current
    build_arg_parser_incremental() interface in configs.py.

    We pass UTC start/end timestamps and map increment_size -> increment-size.
    """
    # Start-of-day UTC
    start_s = f"{d.isoformat()} 00:00:00"

    # Either full day or "smoke" mode with only the first M minutes
    if cfg.minutes_per_day and cfg.minutes_per_day > 0:
        end_dt = datetime.combine(d, datetime.min.time()) + timedelta(minutes=cfg.minutes_per_day)
        end_s = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        next_day = d + timedelta(days=1)
        end_s = f"{next_day.isoformat()} 00:00:00"

    cmd = [
        "python", str(cfg.worker_path),

        # --- time window (UTC) ---
        "--start-dt", start_s,
        "--end-dt",   end_s,

        # --- I/O ---
        "--channels-file", str(cfg.channels_file),
        "--ffl-path",      str(cfg.ffl_path),
        "--ffl-spec",      str(cfg.ffl_spec),
        "--out",           str(cfg.out_dir),

        # --- behavior ---
        "--increment-size", str(cfg.increment_size),
        "--max-retries",    str(cfg.max_retries),
        "--log-level",      cfg.log_level,
    ]

    if cfg.resume:
        cmd.append("--resume")

    # NOTE: we do not pass compression or add_pruned to the worker yet,
    # because the incremental worker parser currently has no such flags.

    return cmd


# =========
# Plan-only
# =========

def plan_only(cfg: ParallelConfig, log: logging.Logger) -> None:
    """Dry run: just mark queued days in the index, no subprocesses."""
    idx = load_index(cfg)
    days = collect_eligible_days(cfg, log)
    for d in days:
        idx = mark_queued(idx, d)
    idx["stats"]["total"] = len(idx.get("days", {}))
    idx["stats"]["queued"] = sum(
        1 for v in idx["days"].values() if v.get("status") == "queued"
    )
    atomic_write_json(index_path(cfg), idx)
    log.info("Dry-run: %d days queued. Nothing spawned.", idx["stats"]["queued"])


# =========
# Scheduler
# =========

def ensure_day_dir(cfg: ParallelConfig, d: date) -> Path:
    p = day_folder(cfg, d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def backoff_delay(attempt: int) -> int:
    # 1st retry 60s, then 120s, clamp to 300s
    return min(300, 60 * (2 ** max(0, attempt - 1)))


def supervise(cfg: ParallelConfig, log: logging.Logger) -> None:
    """
    Main supervisor loop: spawn one worker per day, up to concurrency, with
    retries and optional memory guard.
    """
    idx = load_index(cfg)

    from collections import defaultdict, deque
    queue = deque(collect_eligible_days(cfg, log))
    for d in queue:
        idx = mark_queued(idx, d)
    atomic_write_json(index_path(cfg), idx)

    running: Dict[date, tuple[subprocess.Popen, io.BufferedWriter]] = {}
    retries = defaultdict(int)
    delayed_ready: Dict[date, float] = {}

    stop_spawning = False
    sig_count = 0
    kill_deadline = None  # used to escalate from TERM to KILL

    def terminate_all(kind: str):
        # kind: "TERM" or "KILL"
        for d, (proc, _) in list(running.items()):
            if proc.poll() is None:
                try:
                    if kind == "TERM":
                        log.warning("Sending SIGTERM to %s (pid=%d)", d.isoformat(), proc.pid)
                        proc.terminate()
                    else:
                        log.error("Sending SIGKILL to %s (pid=%d)", d.isoformat(), proc.pid)
                        proc.kill()
                except Exception:
                    pass

    def handle_sig(sig, _frame):
        nonlocal stop_spawning, sig_count
        sig_count += 1
        stop_spawning = True
        if sig_count == 1:
            log.warning(
                "Signal %s received: stop spawning; supervise current jobs to completion. "
                "(Press Ctrl+C again to terminate workers.)",
                sig,
            )
        elif sig_count == 2:
            log.warning(
                "Second interrupt: terminating all running workers (SIGTERM). "
                "(Press Ctrl+C again to force kill.)"
            )
            terminate_all("TERM")
        else:
            log.error(
                "Third interrupt: force-killing all workers (SIGKILL) and exiting."
            )
            terminate_all("KILL")

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    def spawn(d: date) -> bool:
        if not mem_ok(cfg.mem_guard_mb):
            return False
        now = time.time()
        if d in delayed_ready and now < delayed_ready[d]:
            return False
        day_path = ensure_day_dir(cfg, d)
        cmd = build_worker_cmd(cfg, d)
        fout = open(day_path / "worker.outer.log", "ab", buffering=0)
        proc = subprocess.Popen(cmd, stdout=fout, stderr=subprocess.STDOUT)
        running[d] = (proc, fout)
        idx_local = mark_running(idx, d, pid=proc.pid)
        atomic_write_json(index_path(cfg), idx_local)
        log.info("Spawned %s (pid=%d)", d.isoformat(), proc.pid)
        return True

    def maybe_escalate():
        nonlocal kill_deadline
        if sig_count < 2:
            return
        # We already sent SIGTERM; after 10s, escalate to SIGKILL
        if kill_deadline is None:
            kill_deadline = time.time() + 10.0
        if time.time() >= kill_deadline:
            terminate_all("KILL")
            kill_deadline = float("inf")  # avoid repeating

    while queue or running or delayed_ready:
        # Spawn if capacity and allowed
        while (not stop_spawning) and queue and (len(running) < cfg.concurrency):
            picked = None
            now = time.time()
            for _ in range(len(queue)):
                cand = queue[0]
                ready_at = delayed_ready.get(cand, 0.0)
                if now >= ready_at:
                    picked = queue.popleft()
                    break
                queue.rotate(-1)
            if picked is None:
                break
            if spawn(picked):
                time.sleep(cfg.stagger_seconds)
            else:
                queue.appendleft(picked)
                break

        # Poll running jobs
        finished: List[date] = []
        for d, (proc, fout) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            try:
                fout.flush()
            finally:
                fout.close()
            finished.append(d)

            if rc == 0 and day_completed(cfg, d):
                idx = mark_done(idx, cfg, d)
                atomic_write_json(index_path(cfg), idx)
                log.info("Completed %s", d.isoformat())
                delayed_ready.pop(d, None)
                retries.pop(d, None)
            else:
                retries[d] += 1
                if retries[d] <= cfg.max_retries:
                    delay = backoff_delay(retries[d])
                    delayed_ready[d] = time.time() + delay
                    idx = mark_queued(idx, d)
                    atomic_write_json(index_path(cfg), idx)
                    log.warning(
                        "Day %s failed (rc=%s). Retry %d/%d in %ds",
                        d.isoformat(), rc, retries[d], cfg.max_retries, delay,
                    )
                    queue.append(d)
                else:
                    idx = mark_failed(idx, d, error=f"Exit {rc}", retries=retries[d])
                    atomic_write_json(index_path(cfg), idx)
                    log.error(
                        "Day %s failed permanently after %d retries (rc=%s)",
                        d.isoformat(), retries[d], rc,
                    )
                    delayed_ready.pop(d, None)
            running.pop(d, None)

        if not finished:
            maybe_escalate()
            time.sleep(0.5)

        for d in list(delayed_ready.keys()):
            if (d not in queue) and (d not in running):
                delayed_ready.pop(d, None)

        # If we’ve force-killed everything (sig_count >= 3) and nothing is running, bail out
        if sig_count >= 3 and not running:
            break

    # Finalize index stats
    idx = load_index(cfg)
    from utils.parallel_index import _recompute_stats  # local import to avoid exposing it
    _recompute_stats(idx)
    atomic_write_json(index_path(cfg), idx)
    log.info("All done. Stats: %s", json.dumps(idx["stats"]))
