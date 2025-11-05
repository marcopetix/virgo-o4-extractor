from __future__ import annotations

# ======================
# Imports
# ======================
import argparse
import json
import logging
import os
import io
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# ======================
# Dates & Paths
# ======================
from datetime import timedelta
from typing import Dict, List, Tuple
import time
import subprocess
import tempfile

def expand_dates(d0: date, d1: date) -> List[date]:
    """Inclusive [d0, d1] date list."""
    out: List[date] = []
    cur = d0
    while cur <= d1:
        out.append(cur)
        cur = cur + timedelta(days=1)
    return out

def day_dir(out_root: Path, d: date) -> Path:
    return out_root / d.isoformat()

def day_h5_path(out_root: Path, d: date) -> Path:
    ymd = d.strftime("%Y%m%d")
    return day_dir(out_root, d) / f"trend_{ymd}.h5"

def day_summary_path(out_root: Path, d: date) -> Path:
    ymd = d.strftime("%Y%m%d")
    return day_dir(out_root, d) / f"trend_{ymd}.summary.json"

def day_state_path(out_root: Path, d: date) -> Path:
    return day_dir(out_root, d) / "state_day.json"

def day_completed(out_root: Path, d: date) -> bool:
    """A day is completed if summary exists and (optionally) says completed."""
    sp = day_summary_path(out_root, d)
    if not sp.is_file():
        return False
    try:
        j = json.loads(sp.read_text(encoding="utf-8"))
        return str(j.get("status", "completed")).lower() == "completed"
    except Exception:
        # If unreadable but file exists, let worker decide on resume; treat as not completed.
        return False

# ======================
# Global index.json
# ======================

def index_path(out_root: Path) -> Path:
    return out_root / "index.json"

def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)

def load_index(out_root: Path, cfg: ParallelConfig) -> dict:
    p = index_path(out_root)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass  # fall through to fresh
    return {
        "range": {"start_date": str(cfg.start_date), "end_date": str(cfg.end_date)},
        "params": {
            "concurrency": cfg.concurrency,
            "append_interval": cfg.append_interval,
        },
        "days": {},   # date -> {status, retries, ...}
        "stats": {"total": 0, "completed": 0, "failed": 0, "running": 0, "queued": 0},
        "run_utc_started": cfg.run_utc_started,
    }

def _recompute_stats(index: dict) -> None:
    days = index.get("days", {})
    total = len(days)
    completed = sum(1 for v in days.values() if v.get("status") == "completed")
    failed    = sum(1 for v in days.values() if v.get("status") == "failed")
    running   = sum(1 for v in days.values() if v.get("status") == "running")
    queued    = sum(1 for v in days.values() if v.get("status") == "queued")
    index["stats"] = {
        "total": total, "completed": completed, "failed": failed,
        "running": running, "queued": queued
    }

def mark_queued(index: dict, d: date) -> dict:
    days = index.setdefault("days", {})
    rec = days.setdefault(str(d), {})
    rec.update({"status": "queued"})
    _recompute_stats(index)
    return index

def mark_running(index: dict, d: date, pid: int) -> dict:
    days = index.setdefault("days", {})
    rec = days.setdefault(str(d), {})
    rec.update({"status": "running", "pid": pid, "started_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    _recompute_stats(index)
    return index

def mark_done(index: dict, d: date, out_root: Path) -> dict:
    ymd = d.strftime("%Y%m%d")
    rec = index.setdefault("days", {}).setdefault(str(d), {})
    rec.update({
        "status": "completed",
        "pid": None,
        "h5": str(day_h5_path(out_root, d)),
        "summary": str(day_summary_path(out_root, d)),
        "finished_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    _recompute_stats(index)
    return index

def mark_failed(index: dict, d: date, error: str, retries: int) -> dict:
    rec = index.setdefault("days", {}).setdefault(str(d), {})
    rec.update({
        "status": "failed",
        "pid": None,
        "error": error,
        "retries": retries,
        "finished_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    _recompute_stats(index)
    return index

# ======================
# Global index.json
# ======================

def index_path(out_root: Path) -> Path:
    return out_root / "index.json"

def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)

def load_index(out_root: Path, cfg: ParallelConfig) -> dict:
    p = index_path(out_root)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass  # fall through to fresh
    return {
        "range": {"start_date": str(cfg.start_date), "end_date": str(cfg.end_date)},
        "params": {
            "concurrency": cfg.concurrency,
            "append_interval": cfg.append_interval,
        },
        "days": {},   # date -> {status, retries, ...}
        "stats": {"total": 0, "completed": 0, "failed": 0, "running": 0, "queued": 0},
        "run_utc_started": cfg.run_utc_started,
    }

def _recompute_stats(index: dict) -> None:
    days = index.get("days", {})
    total = len(days)
    completed = sum(1 for v in days.values() if v.get("status") == "completed")
    failed    = sum(1 for v in days.values() if v.get("status") == "failed")
    running   = sum(1 for v in days.values() if v.get("status") == "running")
    queued    = sum(1 for v in days.values() if v.get("status") == "queued")
    index["stats"] = {
        "total": total, "completed": completed, "failed": failed,
        "running": running, "queued": queued
    }

def mark_queued(index: dict, d: date) -> dict:
    days = index.setdefault("days", {})
    rec = days.setdefault(str(d), {})
    rec.update({"status": "queued"})
    _recompute_stats(index)
    return index

def mark_running(index: dict, d: date, pid: int) -> dict:
    days = index.setdefault("days", {})
    rec = days.setdefault(str(d), {})
    rec.update({"status": "running", "pid": pid, "started_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    _recompute_stats(index)
    return index

def mark_done(index: dict, d: date, out_root: Path) -> dict:
    ymd = d.strftime("%Y%m%d")
    rec = index.setdefault("days", {}).setdefault(str(d), {})
    rec.update({
        "status": "completed",
        "pid": None,
        "h5": str(day_h5_path(out_root, d)),
        "summary": str(day_summary_path(out_root, d)),
        "finished_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    _recompute_stats(index)
    return index

def mark_failed(index: dict, d: date, error: str, retries: int) -> dict:
    rec = index.setdefault("days", {}).setdefault(str(d), {})
    rec.update({
        "status": "failed",
        "pid": None,
        "error": error,
        "retries": retries,
        "finished_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    _recompute_stats(index)
    return index

# ======================
# Resource guards (optional)
# ======================
def mem_available_mb() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    # value in kB
                    mem_value = int(parts[1]) // 1024
                    log.info(f"Memory available: {mem_value} MB")
                    return mem_value
    except Exception:
        return 1_000_000  # if unknown, pretend plenty
    return 0

def mem_ok(threshold_mb: int) -> bool:
    if threshold_mb <= 0:
        return True
    available_mb = mem_available_mb()
    log.info(f"Memory available: {available_mb} MB (guard threshold: {threshold_mb} MB)")
    return available_mb >= threshold_mb

# ======================
# CLI → Config
# ======================

def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date '{s}' (expected YYYY-MM-DD)") from e


def _positive_int(name: str, v: str) -> int:
    try:
        i = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--{name} must be an integer")
    if i <= 0:
        raise argparse.ArgumentTypeError(f"--{name} must be > 0")
    return i


@dataclass(frozen=True)
class ParallelConfig:
    # Required
    start_date: date
    end_date: date
    channels_file: Path
    ffl_path: Path
    ffl_spec: str
    out_dir: Path
    limit_days: int = 0
    minutes_per_day: int = 0

    # Optional / tuning
    append_interval: int = 100
    concurrency: int = 8
    log_level: str = "INFO"
    resume: bool = True
    max_retries: int = 2
    stagger_seconds: int = 5
    mem_guard_mb: int = 0
    dry_run: bool = False

    # Worker entrypoint (override for tests)
    worker_path: Path = Path("scripts/gwf_to_h5_incremental.py")

    # Derived (filled by normalize_config)
    run_utc_started: str = ""
    args_echo: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gwf_to_h5_parallel.py",
        description="Parallel launcher for daily GWF→HDF5 conversion (one subprocess per day).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

# Required
    p.add_argument("--start-date", type=_parse_date, required=True, help="Inclusive start date (YYYY-MM-DD)")
    p.add_argument("--end-date",   type=_parse_date, required=True, help="Inclusive end date (YYYY-MM-DD)")
    p.add_argument("--channels-file", type=Path, default="channels/o4_channels.txt", help="TXT file with channel names (one per line)")
    p.add_argument("--ffl-path",      type=Path, default="/virgoData/ffl/trend.ffl", help="Frame file list (FFL)")
    p.add_argument("--ffl-spec",      type=str,  default="V1trend", help="FFL spec name (e.g., V1trend)")
    p.add_argument("--out",           dest="out_dir", type=Path, default="/data/procdata/rcsDatasets/OriginalSR/o4_trend_h5", help="Output root directory")

    # Optional
    p.add_argument("--append-interval", type=lambda v: _positive_int("append-interval", v), default=100,
                   help="Append interval in seconds passed to the worker")
    p.add_argument("--concurrency", type=lambda v: _positive_int("concurrency", v), default=8,
                   help="Max concurrent worker processes")
    p.add_argument("--max-retries", type=lambda v: _positive_int("max-retries", v), default=2,
                   help="Max retries per day on nonzero exit")
    p.add_argument("--stagger-seconds", type=lambda v: _positive_int("stagger-seconds", v), default=5,
                   help="Delay between spawns to avoid NFS bursts")
    p.add_argument("--mem-guard-mb", type=int, default=0,
                   help="If >0, do not spawn new jobs when MemAvailable < this many MB")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARN", "ERROR"], default="INFO")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Pass --resume to workers and skip fully completed days")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Plan only; do not spawn workers")

    # Testing/override
    p.add_argument("--worker-path", type=Path, default=Path("scripts/gwf_to_h5_incremental.py"),
                   help="Path to the single-day worker script")
    p.add_argument("--limit-days", type=int, default=0,
                help="Limit how many days to schedule (0 = no limit).")
    p.add_argument("--minutes-per-day", type=int, default=0,
                help="Extract only the first M minutes of each day (0 = full day).")


    return p.parse_args()

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    # Ensure UTC timestamps in logs
    logging.Formatter.converter = lambda *args: datetime.utcnow().timetuple()


def normalize_config(ns: argparse.Namespace) -> ParallelConfig:
    # Validate date range
    if ns.start_date > ns.end_date:
        raise SystemExit("start-date must be <= end-date")

    # Validate files/dirs
    if not ns.channels_file.is_file():
        raise SystemExit(f"channels file not found: {ns.channels_file}")
    if not ns.ffl_path.is_file():
        raise SystemExit(f"ffl file not found: {ns.ffl_path}")

    out_dir: Path = ns.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    worker_path: Path = ns.worker_path
    if not worker_path.exists():
        raise SystemExit(f"worker script not found: {worker_path}")

    # Build config
    cfg = ParallelConfig(
        start_date=ns.start_date,
        end_date=ns.end_date,
        channels_file=ns.channels_file.resolve(),
        ffl_path=ns.ffl_path.resolve(),
        ffl_spec=str(ns.ffl_spec),
        out_dir=out_dir.resolve(),
        append_interval=int(ns.append_interval),
        concurrency=int(ns.concurrency),
        log_level=str(ns.log_level),
        resume=bool(ns.resume),
        max_retries=int(ns.max_retries),
        stagger_seconds=int(ns.stagger_seconds),
        mem_guard_mb=int(ns.mem_guard_mb),
        dry_run=bool(ns.dry_run),
        worker_path=worker_path.resolve(),
        run_utc_started=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        args_echo=" ".join(os.sys.argv),
        limit_days=ns.limit_days,
        minutes_per_day=ns.minutes_per_day,
    )

    # Validate limits
    if cfg.limit_days < 0:
        raise SystemExit("--limit-days must be >= 0")
    if cfg.minutes_per_day < 0:
        raise SystemExit("--minutes-per-day must be >= 0")
    if cfg.minutes_per_day and cfg.minutes_per_day < cfg.append_interval // 60:
        # keep it simple: we want at least one full chunk
        raise SystemExit("--minutes-per-day must be >= append_interval/60")
    
    return cfg


def echo_run_header(cfg: ParallelConfig, log: logging.Logger) -> None:
    payload = {
        "range": {"start_date": str(cfg.start_date), "end_date": str(cfg.end_date)},
        "concurrency": cfg.concurrency,
        "append_interval": cfg.append_interval,
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
    }
    log.info("===== gwf_to_h5_parallel: start =====")
    log.info(json.dumps(payload, indent=2))

# ======================
# Main (plan preview)
# ======================
def collect_eligible_days(cfg: ParallelConfig, log: logging.Logger) -> list[date]:
    all_days = expand_dates(cfg.start_date, cfg.end_date)
    eligible: list[date] = []
    for d in all_days:
        if cfg.resume and day_completed(cfg.out_dir, d):
            log.info("Skip (already completed): %s", d.isoformat())
            continue
        eligible.append(d)
    if cfg.limit_days and len(eligible) > cfg.limit_days:
        log.info("Limiting days: taking first %d of %d eligible", cfg.limit_days, len(eligible))
        eligible = eligible[:cfg.limit_days]
    return eligible



def plan_only(cfg: ParallelConfig, log: logging.Logger) -> None:
    idx = load_index(cfg.out_dir, cfg)
    days = collect_eligible_days(cfg, log)
    for d in days:
        idx = mark_queued(idx, d)
    idx["stats"]["total"] = len(idx.get("days", {}))
    idx["stats"]["queued"] = sum(1 for v in idx["days"].values() if v.get("status") == "queued")
    atomic_write_json(index_path(cfg.out_dir), idx)
    log.info("Dry-run: %d days queued. Nothing spawned.", idx["stats"]["queued"])

# ======================
# Worker command
# ======================
# REPLACE your build_worker_cmd with this version
from datetime import timedelta

def build_worker_cmd(cfg: ParallelConfig, d: date) -> list[str]:
    start_s = f"{d.isoformat()} 00:00:00"
    if cfg.minutes_per_day and cfg.minutes_per_day > 0:
        # Smoke mode: run only the first M minutes of the day
        end_dt = datetime.combine(d, datetime.min.time()) + timedelta(minutes=cfg.minutes_per_day)
        end_s = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        # Full day
        next_day = d + timedelta(days=1)
        end_s   = f"{next_day.isoformat()} 00:00:00"

    cmd = [
        "python", str(cfg.worker_path),
        "--start", start_s,
        "--end",   end_s,
        "--channels-file", str(cfg.channels_file),
        "--ffl-path", str(cfg.ffl_path),
        "--ffl-spec", str(cfg.ffl_spec),
        "--out", str(cfg.out_dir),
        "--append-interval", str(cfg.append_interval),
        "--log-level", cfg.log_level,
    ]
    if cfg.resume:
        cmd.append("--resume")
    return cmd


# ======================
# Scheduler
# ======================
import signal
import time
import subprocess
from collections import defaultdict, deque

def ensure_day_dir(root: Path, d: date) -> Path:
    p = day_dir(root, d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def backoff_delay(attempt: int) -> int:
    # 1st retry 60s, then 120s, clamp to 300s
    return min(300, 60 * (2 ** max(0, attempt - 1)))


def supervise(cfg: ParallelConfig, log: logging.Logger) -> None:
    idx = load_index(cfg.out_dir, cfg)

    queue = deque(collect_eligible_days(cfg, log))
    for d in queue:
        idx = mark_queued(idx, d)
    atomic_write_json(index_path(cfg.out_dir), idx)

    running: dict[date, tuple[subprocess.Popen, "io.BufferedWriter"]] = {}
    retries: defaultdict[date, int] = defaultdict(int)
    delayed_ready: dict[date, float] = {}

    stop_spawning = False
    sig_count = 0

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
            log.warning("Signal %s received: stop spawning; supervise current jobs to completion. (Press Ctrl+C again to terminate workers.)", sig)
        elif sig_count == 2:
            log.warning("Second interrupt: terminating all running workers (SIGTERM). (Press Ctrl+C again to force kill.)")
            terminate_all("TERM")
        else:
            log.error("Third interrupt: force-killing all workers (SIGKILL) and exiting.")
            terminate_all("KILL")

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    def spawn(d: date):
        if not mem_ok(cfg.mem_guard_mb):
            return False
        now = time.time()
        if d in delayed_ready and now < delayed_ready[d]:
            return False
        ensure_day_dir(cfg.out_dir, d)
        cmd = build_worker_cmd(cfg, d)
        fout = open(day_dir(cfg.out_dir, d) / "worker.log", "ab", buffering=0)
        proc = subprocess.Popen(cmd, stdout=fout, stderr=subprocess.STDOUT)
        running[d] = (proc, fout)
        idx_local = mark_running(idx, d, pid=proc.pid)
        atomic_write_json(index_path(cfg.out_dir), idx_local)
        log.info("Spawned %s (pid=%d)", d.isoformat(), proc.pid)
        return True

    kill_deadline = None  # if we issued SIGTERM, set a deadline to escalate

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
        finished: list[date] = []
        for d, (proc, fout) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            try:
                fout.flush()
            finally:
                fout.close()
            finished.append(d)

            if rc == 0 and day_completed(cfg.out_dir, d):
                idx = mark_done(idx, d, cfg.out_dir)
                atomic_write_json(index_path(cfg.out_dir), idx)
                log.info("Completed %s", d.isoformat())
                delayed_ready.pop(d, None)
                retries.pop(d, None)
            else:
                retries[d] += 1
                if retries[d] <= cfg.max_retries:
                    delay = backoff_delay(retries[d])
                    delayed_ready[d] = time.time() + delay
                    idx = mark_queued(idx, d)
                    atomic_write_json(index_path(cfg.out_dir), idx)
                    log.warning("Day %s failed (rc=%s). Retry %d/%d in %ds",
                                d.isoformat(), rc, retries[d], cfg.max_retries, delay)
                    queue.append(d)
                else:
                    idx = mark_failed(idx, d, error=f"Exit {rc}", retries=retries[d])
                    atomic_write_json(index_path(cfg.out_dir), idx)
                    log.error("Day %s failed permanently after %d retries (rc=%s)", d.isoformat(), retries[d], rc)
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

    idx = load_index(cfg.out_dir, cfg)
    _recompute_stats(idx)
    atomic_write_json(index_path(cfg.out_dir), idx)
    log.info("All done. Stats: %s", json.dumps(idx["stats"]))


if __name__ == "__main__":
    ns = parse_args()
    setup_logging(ns.log_level)
    log = logging.getLogger("parallel")
    cfg = normalize_config(ns)
    echo_run_header(cfg, log)

    if cfg.dry_run:
        plan_only(cfg, log)
    else:
        supervise(cfg, log)