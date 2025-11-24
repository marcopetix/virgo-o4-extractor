from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict

from utils.configs import ParallelConfig


def day_folder_name(ffl_spec: str, d: date) -> str:
    """Name of the per-day folder, matching the worker."""
    return f"{ffl_spec}-{d.isoformat()}"  # e.g. V1trend-2024-04-14


def day_folder(cfg: ParallelConfig, d: date) -> Path:
    """Per-day folder, matching worker's _build_output_paths."""
    return cfg.out_dir / day_folder_name(cfg.ffl_spec, d)


def day_h5_path(cfg: ParallelConfig, d: date) -> Path:
    """Per-day HDF5 path, matching worker naming."""
    ymd = d.strftime("%Y%m%d")
    return day_folder(cfg, d) / f"{cfg.ffl_spec}_{ymd}.h5"


def day_summary_path(cfg: ParallelConfig, d: date) -> Path:
    """Per-day general summary JSON path, matching worker naming."""
    ymd = d.strftime("%Y%m%d")
    return day_folder(cfg, d) / f"{cfg.ffl_spec}_{ymd}.summary.json"


def day_state_path(cfg: ParallelConfig, d: date) -> Path:
    """Per-day state file path, as written by the worker."""
    return day_folder(cfg, d) / "state_day.json"


def day_completed(cfg: ParallelConfig, d: date) -> bool:
    """
    A day is considered completed if state_day.json exists and its status
    is 'completed'. This aligns with the worker's update_day_state().
    """
    sp = day_state_path(cfg, d)
    if not sp.is_file():
        return False
    try:
        state = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(state.get("status", "")).lower() == "completed"


def index_path(cfg: ParallelConfig) -> Path:
    """Global index for the whole run."""
    return cfg.out_dir / "index.json"


def atomic_write_json(path: Path, obj: dict) -> None:
    """Simple atomic JSON writer (write to tmp then replace)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_index(cfg: ParallelConfig) -> dict:
    """
    Load index.json if present, otherwise initialize a fresh index structure.
    """
    p = index_path(cfg)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass  # fall through to fresh

    return {
        "range": {
            "start_date": str(cfg.start_date),
            "end_date": str(cfg.end_date),
        },
        "params": {
            "concurrency": cfg.concurrency,
            "increment_size": cfg.increment_size,
        },
        "days": {},   # date -> {status, retries, ...}
        "stats": {"total": 0, "completed": 0, "failed": 0, "running": 0, "queued": 0},
        "run_utc_started": cfg.run_utc_started,
    }


def _recompute_stats(index: dict) -> None:
    days = index.get("days", {})
    total = len(days)
    completed = sum(1 for v in days.values() if v.get("status") == "completed")
    failed = sum(1 for v in days.values() if v.get("status") == "failed")
    running = sum(1 for v in days.values() if v.get("status") == "running")
    queued = sum(1 for v in days.values() if v.get("status") == "queued")
    index["stats"] = {
        "total": total,
        "completed": completed,
        "failed": failed,
        "running": running,
        "queued": queued,
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
    rec.update({
        "status": "running",
        "pid": pid,
        "started_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    _recompute_stats(index)
    return index


def mark_done(index: dict, cfg: ParallelConfig, d: date) -> dict:
    rec = index.setdefault("days", {}).setdefault(str(d), {})
    rec.update({
        "status": "completed",
        "pid": None,
        "h5": str(day_h5_path(cfg, d)),
        "summary": str(day_summary_path(cfg, d)),
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
