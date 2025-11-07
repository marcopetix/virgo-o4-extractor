from __future__ import annotations

# =========
# Imports
# =========
import argparse
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import logging
import sys
import signal
import time
import json
import os
from pathlib import Path
import subprocess
from typing import Optional, List, Tuple, Set, Dict, Any
from collections import defaultdict
import tempfile
import numpy as np
import h5py



# =========
# CLI → Config
# =========

ISO_TS_HINT = 'YYYY-MM-DD HH:MM:SS'  # e.g., "2025-08-10 00:00:00"

def _parse_utc(ts: str) -> datetime:
    """
    Parse a timestamp like 'YYYY-MM-DD HH:MM:SS' and return an aware datetime in UTC.
    We assume inputs are intended as UTC (trend data is GPS/UTC aligned).
    """
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid timestamp '{ts}'. Expected format {ISO_TS_HINT}"
        ) from e
    return dt.replace(tzinfo=timezone.utc)

def _str_date_folder(dt: datetime) -> str:
    return dt.date().isoformat()  # 'YYYY-MM-DD'

@dataclass(frozen=True)
class Config:
    # Core time window (UTC)
    start_dt: datetime
    end_dt: datetime
    append_interval: int  # seconds (e.g., 100)

    # I/O inputs
    channels_file: Path
    ffl_path: Path
    ffl_spec: str

    # Outputs (resolved paths)
    out_root: Path           # base output directory
    day_folder: Path         # out_root/YYYY-MM-DD/
    h5_path: Path            # day_folder/trend_YYYYmmdd.h5
    state_path: Path         # day_folder/state_day.json
    summary_path: Path       # day_folder/trend_YYYYmmdd.summary.json
    validation_path: Path    # day_folder/validation.json
    log_path: Path           # day_folder/worker.log

    # Behavior flags
    resume: bool
    dry_run: bool
    log_level: str  # 'INFO' | 'DEBUG'

    # Compression policy
    compression: str 
    gzip_level: int = 1 if compression == "gzip" else 0

    # Fixed policies for v1 (no CLI flags)
    # We still keep them in config for clarity & future-proofing.
    dtype_policy: str = dataclasses.field(default="keep")  # fixed

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gwf_to_h5_incremental.py",
        description=(
            "Convert Virgo trend GWF → daily HDF5 using a Growing Jars strategy "
            "(single-day worker)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required time window
    p.add_argument("--start", required=True, type=_parse_utc,
                   help=f"Start timestamp in UTC, like {ISO_TS_HINT}")
    p.add_argument("--end", required=True, type=_parse_utc,
                   help=f"End timestamp in UTC, like {ISO_TS_HINT}")

    # Required I/O inputs
    p.add_argument("--channels-file", type=Path,
                   help="TXT file with one channel per line ('#' for comments).", default="channels/o4_channels.txt")
    p.add_argument("--ffl-path", type=Path,
                   help="Path to the FFL file (e.g., /virgoData/ffl/trend.ffl).", default="/virgoData/ffl/trend.ffl")
    p.add_argument("--ffl-spec", type=str,
                   help="FFL spec name (e.g., V1trend).", default="V1trend")

    # Output root
    p.add_argument("--out", dest="out_root", type=Path,
                   help="Root output directory where the day folder will be created.", default="/data/procdata/rcsDatasets/OriginalSR/o4_trend_h5")
    p.add_argument("--compression", type=str, choices=["gzip", "lzf", "none"], default="gzip",
                   help="Compression method for the output HDF5 file.")

    # Behavior
    p.add_argument("--append-interval", type=int, default=100,
                   help="Chunk duration in seconds (aligned to frames).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from state_day.json if present.")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan & validate; do not write HDF5.")
    p.add_argument("--log-level", choices=["INFO", "DEBUG"], default="INFO",
                   help="Logging verbosity.")

    return p

def _ensure_inputs_exist(args: argparse.Namespace) -> None:
    if not args.channels_file.is_file():
        raise FileNotFoundError(f"channels file not found: {args.channels_file}")
    if not args.ffl_path.is_file():
        raise FileNotFoundError(f"FFL path not found: {args.ffl_path}")
    if args.append_interval <= 0:
        raise ValueError("--append-interval must be a positive integer (seconds)")
    if args.start >= args.end:
        raise ValueError("--start must be earlier than --end")
    
def _mk_day_paths(out_root: Path, start_dt: datetime) -> tuple[Path, Path, Path, Path, Path]:
    day_folder = out_root / _str_date_folder(start_dt)
    ymd = start_dt.strftime("%Y%m%d")
    h5_path = day_folder / f"trend_{ymd}.h5"
    summary_path = day_folder / f"trend_{ymd}.summary.json"
    state_path = day_folder / "state_day.json"
    validation_path = day_folder / "validation.json"
    log_path = day_folder / "worker.log"
    return day_folder, h5_path, summary_path, state_path, validation_path, log_path

def build_config(args: argparse.Namespace) -> Config:
    _ensure_inputs_exist(args)

    # Resolve/normalize paths (out_root may not exist yet; created later)
    out_root = args.out_root.resolve()
    day_folder, h5_path, summary_path, state_path, validation_path, log_path = _mk_day_paths(
        out_root, args.start
    )

    # Note: We don’t create directories here; that happens in the next step.
    return Config(
        start_dt=args.start,
        end_dt=args.end,
        append_interval=args.append_interval,
        channels_file=args.channels_file.resolve(),
        ffl_path=args.ffl_path.resolve(),
        ffl_spec=args.ffl_spec,
        out_root=out_root,
        day_folder=day_folder,
        h5_path=h5_path,
        state_path=state_path,
        summary_path=summary_path,
        validation_path=validation_path,
        log_path=log_path,
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        log_level=args.log_level,
        # Fixed policies for v1:
        compression=args.compression,
        gzip_level=1 if args.compression == "gzip" else 0,
        dtype_policy="keep",
    )

# Logging and folder setup

def ensure_day_folder(cfg: Config) -> None:
    """
    Create the day folder (OUT_ROOT/YYYY-MM-DD) if it doesn't exist.
    This function does NOT create the HDF5 file itself.
    """
    cfg.day_folder.mkdir(parents=True, exist_ok=True)

def setup_logging(cfg: Config) -> logging.Logger:
    """
    Set up console + file logging. Timestamps are in UTC.
    """
    logger = logging.getLogger("gwf_to_h5")
    logger.setLevel(logging.DEBUG)  # capture everything; handlers will filter

    # Avoid duplicate handlers if re-run in the same interpreter
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    # Common formatter (UTC time)
    fmt = logging.Formatter(
        fmt="%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Make formatter use UTC
    fmt.converter = time.gmtime  # type: ignore[attr-defined]

    # Console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG if cfg.log_level == "DEBUG" else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (in day folder)
    fh = logging.FileHandler(cfg.log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # keep full detail in file
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Small header with key config fields
    logger.info("===== gwf_to_h5_incremental: start =====")
    logger.info(
        "Window: %s → %s | append_interval=%ss | resume=%s | dry_run=%s | level=%s",
        cfg.start_dt.isoformat(), cfg.end_dt.isoformat(),
        cfg.append_interval, cfg.resume, cfg.dry_run, cfg.log_level,
    )
    logger.info(
        "Channels: %s | FFL: [%s] %s | Out: %s",
        cfg.channels_file, cfg.ffl_spec, cfg.ffl_path, cfg.day_folder,
    )
    logger.debug("Policies: compression=%s(level=%d) dtype_policy=%s",
                 cfg.compression, cfg.gzip_level, cfg.dtype_policy)
    return logger

# =======================
# Atomic JSON utilities
# =======================

def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """
    Write JSON to 'path' atomically: write to a temp file in the same directory then rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)  # atomic on POSIX
    except Exception:
        # Best effort to cleanup on failure
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


def load_channels(channels_file: Path) -> List[str]:
    """
    Load channel names from a TXT file.
    - Strips whitespace
    - Ignores blank lines and lines starting with '#'
    - De-duplicates while preserving order
    """
    seen = set()
    out: List[str] = []
    with channels_file.open("r", encoding="utf-8") as f:
        for line in f:
            # strip comments (allow inline '# ...')
            raw = line.strip()
            if not raw:
                continue
            # If there's an inline comment, cut it
            if "#" in raw:
                raw = raw.split("#", 1)[0].strip()
            if not raw:
                continue
            if raw not in seen:
                seen.add(raw)
                out.append(raw)
    if not out:
        raise ValueError(f"No channels found in {channels_file}")
    return out

def build_chunk_plan(start_dt: datetime, end_dt: datetime, step_seconds: int) -> List[Tuple[datetime, datetime]]:
    """
    Generate [t0, t1) intervals of length 'step_seconds' covering [start_dt, end_dt).
    The last chunk is dropped if it would exceed 'end_dt' (strict coverage).
    """
    plan: List[Tuple[datetime, datetime]] = []
    cur = start_dt
    delta = timedelta(seconds=step_seconds)  # needs timedelta import
    while True:
        t1 = cur + delta
        if t1 > end_dt:
            break
        plan.append((cur, t1))
        cur = t1
    return plan

# ===== Validation constants (v1 simple) =====
VALIDATION_SCAN_LIMIT = 3         # uniformly spread frames to scan
FRCHANNELS_PATH = "FrChannels"    # shell-invoked tool
ASSUMED_TREND_SAMPLE_RATE = 1.0   # 1 Hz trends

GPS_EPOCH_UTC = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
# GPS-UTC offset has been 18s since 2017-01-01. Keep it simple for O4.
GPS_LEAP_SECONDS = 18

def utc_to_gps(dt: datetime) -> float:
    """Convert aware UTC datetime → GPS seconds (simple, fixed 18s offset)."""
    if dt.tzinfo is None:
        raise ValueError("utc_to_gps requires an aware UTC datetime")
    return (dt - GPS_EPOCH_UTC).total_seconds() + GPS_LEAP_SECONDS

# =========================
# channel validation helpers (optimized)
# =========================

def parse_ffl(ffl_path: Path) -> List[Tuple[str, float, float]]:
    """Return list of (file, t0_gps, t1_gps) from an FFL (best-effort)."""
    entries: List[Tuple[str, float, float]] = []
    with ffl_path.open("r", encoding="utf-8") as f:
        for line in f:
            cols = line.split()
            if len(cols) < 3:
                continue
            try:
                file = cols[0]
                t0 = float(cols[1])
                dur = float(cols[2])
                entries.append((file, t0, t0 + dur))
            except Exception:
                # skip malformed lines silently
                continue
    return entries

def overlapping_files(ffl_entries: List[Tuple[str, float, float]],
                      t0_gps: float, t1_gps: float) -> List[str]:
    """Filter FFL entries that overlap [t0_gps, t1_gps)."""
    return [f for (f, a, b) in ffl_entries if not (b <= t0_gps or a >= t1_gps)]

def _uniform_sample(items: List[str], k: int) -> List[str]:
    """Pick k items roughly uniformly across the list."""
    if k <= 0 or not items:
        return []
    if k >= len(items):
        return items
    # spread indices across [0, len-1]
    step = (len(items) - 1) / (k - 1)
    idxs = [round(i * step) for i in range(k)]
    # de-dup in case rounding collides
    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    # If collisions reduced count, pad from remaining
    if len(out) < k:
        for i, _ in enumerate(items):
            if i not in seen:
                out.append(items[i])
                if len(out) == k:
                    break
    return out

def frchannels_list(frchannels_path: str, file_path: str) -> Set[str]:
    """Run FrChannels on a single frame file and return a set of channel names."""
    attempts = (
        [frchannels_path, "-l", file_path],
        [frchannels_path, file_path],
    )
    last_err = None
    for cmd in attempts:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
            chans: Set[str] = set()
            for line in out.splitlines():
                s = line.strip()
                if not s or s.startswith(("Usage:", "This utility")):
                    continue
                name = s.split()[0]
                if name:
                    chans.add(name)
            if chans:
                return chans
        except subprocess.CalledProcessError as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("FrChannels invocation failed")

def validate_channels(
    ffl_path: Path,
    start_gps: float,
    end_gps: float,
    requested: List[str],
    frchannels_path: str = FRCHANNELS_PATH,
    scan_limit: int = VALIDATION_SCAN_LIMIT,
    log: Optional[logging.Logger] = None,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Centralized validation: scan a limited number of overlapping frames with FrChannels and
    return (valid_list, missing_list, details_dict) preserving requested order.
    Also returns a small details dict with scanned files and union size.
    """
    entries = parse_ffl(ffl_path)
    files = overlapping_files(entries, start_gps, end_gps)

    # Choose up to 'scan_limit' files, uniformly spread across the day window.
    files_to_scan = _uniform_sample(files, scan_limit)
    inventory: Set[str] = set()
    scanned: List[str] = []

    for fp in files_to_scan:
        if Path(fp).is_file():
            try:
                inv = frchannels_list(frchannels_path, fp)
                inventory |= inv
                scanned.append(fp)
                if log:
                    log.debug("Scanned %s: +%d (union=%d)", fp, len(inv), len(inventory))
            except Exception as e:
                if log:
                    log.warning("FrChannels failed on %s: %s", fp, e)
        else:
            if log:
                log.debug("Skipping missing frame %s", fp)

    # preserve request order
    req_ordered = list(dict.fromkeys(requested))
    valid = [ch for ch in req_ordered if ch in inventory]
    missing = [ch for ch in req_ordered if ch not in inventory]

    details = {
        "scanned_files": scanned,
        "inventory_size": len(inventory),
        "scan_limit": scan_limit,
        "ffl_entries_considered": len(files),
    }
    return valid, missing, details

def quick_probe_read(
    start_dt: datetime,
    end_dt: datetime,
    channels: List[str],
    ffl_spec: str,
    ffl_path: Path,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {"ok": False, "read_channels": 0, "error": None}
    try:
        from gwdama.io import GwDataManager  # type: ignore
    except Exception as e:
        info["error"] = f"gwdama unavailable: {e}"
        if log:
            log.debug("Probe skipped: %s", info["error"])
        return info

    t0 = start_dt
    t1 = min(start_dt + timedelta(seconds=100), end_dt)
    if t1 <= t0:
        info["error"] = "Empty window for probe"
        return info

    start_gps = float(utc_to_gps(t0))
    end_gps   = float(utc_to_gps(t1))

    try:
        try:
            dm = GwDataManager()
            result = dm.read_gwdata(
                start=start_gps, end=end_gps, channels=channels,
                ffl_spec=ffl_spec, ffl_path=str(ffl_path),
                return_output=True
            )
        except TypeError:
            result = GwDataManager.read_gwdata(
                t0=start_gps, t1=end_gps, channels=channels,
                ffl_spec=ffl_spec, ffl_path=str(ffl_path),
                return_output=True
            )

        data = result[0] if isinstance(result, tuple) else result
        if isinstance(data, dict):
            info["read_channels"] = sum(1 for v in data.values() if v is not None)
        info["ok"] = True
        if log:
            log.info("Probe read ok: %d/%d channels returned in first chunk",
                     info["read_channels"], len(channels))
    except Exception as e:
        info["error"] = str(e)
        if log:
            log.warning("Probe read failed: %s", e)
    return info


def write_validation_json(cfg: Config, channels: List[str], log: logging.Logger) -> None:
    start_gps = utc_to_gps(cfg.start_dt)
    end_gps = utc_to_gps(cfg.end_dt)

    valid, missing, details = validate_channels(
        ffl_path=cfg.ffl_path,
        start_gps=start_gps,
        end_gps=end_gps,
        requested=channels,
        scan_limit=VALIDATION_SCAN_LIMIT,
        log=log,
    )

    probe = quick_probe_read(
        start_dt=cfg.start_dt,
        end_dt=cfg.end_dt,
        channels=channels,
        ffl_spec=cfg.ffl_spec,
        ffl_path=cfg.ffl_path,
        log=log,
    )

    payload = {
        "date": cfg.start_dt.date().isoformat(),
        "start_utc": cfg.start_dt.isoformat(),
        "end_utc": cfg.end_dt.isoformat(),
        "start_gps": start_gps,
        "end_gps": end_gps,
        "append_interval_s": cfg.append_interval,
        "channels_requested": len(channels),
        "channels_valid": len(valid),
        "channels_missing": len(missing),
        "valid_channels": valid,     
        "missing_channels": missing, 
        "sample_rate_hz": ASSUMED_TREND_SAMPLE_RATE,
        "ffl_spec": cfg.ffl_spec,
        "ffl_path": str(cfg.ffl_path),
        "validation_method": {
            "type": "frchannels_uniform_sample",
            "scan_limit": VALIDATION_SCAN_LIMIT,
            **details,
        },
        "probe": probe,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policies": {
            "compression": cfg.compression,
            "gzip_level": cfg.gzip_level,
            "dtype_policy": cfg.dtype_policy,
        },
    }

    # Write atomically
    tmp = cfg.validation_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp.replace(cfg.validation_path)

    log.info(
        "Validation: %d/%d channels seen in frames sample; missing ~%d (see %s)",
        len(valid), len(channels), len(missing), cfg.validation_path
    )

def _gwdama_timestamp_now() -> float:
    """Return current time as gwdama timestamp (GPS seconds)."""
    now_utc = datetime.now(timezone.utc)
    return utc_to_gps(now_utc)

# ============================================================
# HDF5 Writer — gwdama-compatible semantics (ragged, no NaN pad)
# ============================================================

class DayAppendH5Writer:
    """
    gwdama-like writer:
      - /channels/<channel> datasets
      - attrs: channel, sample_rate (Hz), t0 (GPS), unit (optional)
      - append only the samples we have; no NaN padding for gaps
    """

    def __init__(
        self,
        path: Path,
        start_utc: datetime,
        end_utc: datetime,
        append_len: int,
        sample_rate_hz: float = 1.0,
        compression: str = "gzip",
        gzip_level: int = 1,
        dtype_policy: str = "keep",
        log: Optional[logging.Logger] = None,
        # no all_channels, no padding flags – pure gwdama semantics
    ):
        self.path = Path(path)
        self.append_len = int(append_len)
        self.sample_rate_hz = float(sample_rate_hz)
        self.compression = compression
        self.gzip_level = int(gzip_level)
        self.dtype_policy = dtype_policy
        self.log = log

        self.start_utc = start_utc
        self.end_utc = end_utc

        self._h5: Optional[h5py.File] = None
        self._channels_grp: Optional[h5py.Group] = None
        self._dsets: Dict[str, h5py.Dataset] = {}
        self._first_written: Dict[str, bool] = {}   # to stamp t0 once
        self._dtype_by_ch: Dict[str, np.dtype] = {} # remember chosen dtype

        self._file_attrs = {
            "start_utc": start_utc.isoformat(),
            "end_utc": end_utc.isoformat(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "append_len_samples": self.append_len,
            "sample_rate_hz": self.sample_rate_hz,
            "strategy": "resizable-append-1d (ragged, gwdama-like)",
            "compression": f"{self.compression}-{self.gzip_level}",
            "dtype_policy": self.dtype_policy,
        }

    # ---- lifecycle ----

    def open(self) -> None:
        self._h5 = h5py.File(self.path, "a")
        # stamp generic attrs
        for k, v in self._file_attrs.items():
            try:
                self._h5.attrs[k] = v
            except Exception:
                pass
        # gwdama-ish header (matches your sample style)
        try:
            self._h5.attrs["dama_name"] = "gwf2h5_incremental"
            self._h5.attrs["time_stamp"] = _gwdama_timestamp_now()
        except Exception:
            pass
        # group
        try:
            self._channels_grp = self._h5.require_group("channels")
        except Exception:
            self._channels_grp = None
        if self.log:
            self.log.info("Opened HDF5 for append: %s", self.path)

    def close(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.flush()
            finally:
                self._h5.close()
            if self.log:
                self.log.info("Closed HDF5: %s", self.path)
        self._h5 = None
        self._channels_grp = None
        self._dsets.clear()
        self._first_written.clear()
        self._dtype_by_ch.clear()

    # ---- internals ----

    def _parent(self) -> h5py.Group:
        if self._h5 is None:
            raise RuntimeError("HDF5 file is not open")
        return self._channels_grp or self._h5

    def _effective_dtype(self, ch: str, arr: Optional[np.ndarray]) -> np.dtype:
        # keep dtype as read on first sight, else reuse
        if ch in self._dtype_by_ch:
            return self._dtype_by_ch[ch]
        dt = np.asarray(arr).dtype if isinstance(arr, np.ndarray) else np.dtype("float32")
        self._dtype_by_ch[ch] = dt
        return dt

    # ---- dataset handling ----

    def ensure_dataset(
        self,
        ch_name: str,
        first_chunk_t0_utc: Optional[datetime] = None,
        unit: Optional[str] = None,
        sample_rate_hz: Optional[float] = None,
        example_array: Optional[np.ndarray] = None,
    ) -> h5py.Dataset:
        """
        Ensure resizable dataset /channels/<ch_name> exists.
        On resume, if it already exists in the file, re-open it instead of creating.
        """
        if self._h5 is None:
            raise RuntimeError("HDF5 file is not open")

        # In-memory cache hit
        if ch_name in self._dsets:
            return self._dsets[ch_name]

        parent = self._parent()

        # >>> NEW: reuse existing dataset on resume <<<
        if ch_name in parent:
            dset = parent[ch_name]
            # cache & first-written flag based on presence of 't0'
            self._dsets[ch_name] = dset
            self._first_written[ch_name] = "t0" in dset.attrs

            # backfill attrs if missing
            try:
                if "channel" not in dset.attrs:
                    dset.attrs["channel"] = ch_name
                if "sample_rate" not in dset.attrs and sample_rate_hz is not None:
                    dset.attrs["sample_rate"] = float(sample_rate_hz)
                if unit is not None and "unit" not in dset.attrs:
                    dset.attrs["unit"] = str(unit)
            except Exception:
                pass

            if self.log:
                self.log.debug("Reusing existing dataset for '%s' (resume)", ch_name)
            return dset

        # Create new dataset
        dt = self._effective_dtype(ch_name, example_array)
        dset = parent.create_dataset(
            name=ch_name,
            shape=(0,),
            maxshape=(None,),
            dtype=dt,
            chunks=(self.append_len,),
            compression=self.compression,
            compression_opts=self.gzip_level,
            shuffle=False,
            fletcher32=False,
        )

        # Set attrs
        sr = float(sample_rate_hz if sample_rate_hz is not None else self.sample_rate_hz)
        try:
            dset.attrs["channel"] = ch_name
            dset.attrs["sample_rate"] = sr
            if unit is not None:
                dset.attrs["unit"] = str(unit)
            # 't0' is stamped on first successful append
        except Exception:
            pass

        self._dsets[ch_name] = dset
        self._first_written[ch_name] = False
        return dset

    # ---- append & flush ----

    def _write_block(self, dset: h5py.Dataset, block: np.ndarray) -> int:
        old = int(dset.shape[0])
        n = int(block.shape[0])
        dset.resize((old + n,))
        dset[old:old + n] = block.astype(dset.dtype, copy=False)
        return n

    def append_chunk(
        self,
        data_by_ch: Dict[str, np.ndarray],
        meta_by_ch: Optional[Dict[str, Dict[str, Any]]] = None,
        chunk_t0_utc: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """
        Append one window for channels present in data_by_ch (no NaN padding).
        Returns per-channel written sample counts.
        """
        if self._h5 is None:
            raise RuntimeError("HDF5 file is not open")

        written: Dict[str, int] = {}

        for ch, arr in data_by_ch.items():
            if not isinstance(arr, np.ndarray):
                continue
            md = (meta_by_ch or {}).get(ch, {})
            sr = float(md.get("sample_rate", self.sample_rate_hz))
            unit = md.get("unit", None)

            dset = self.ensure_dataset(
                ch_name=ch,
                first_chunk_t0_utc=chunk_t0_utc,
                unit=unit,
                sample_rate_hz=sr,
                example_array=arr,
            )

            # stamp t0 on FIRST successful write only
            if not self._first_written.get(ch, False) and chunk_t0_utc is not None:
                if "t0" not in dset.attrs:  # <— added guard
                    dset.attrs["t0"] = float(utc_to_gps(chunk_t0_utc))
                self._first_written[ch] = True

            block = np.asarray(arr)
            written[ch] = self._write_block(dset, block)

        return written

    def flush(self) -> None:
        if self._h5 is not None:
            self._h5.flush()

    # ---- summary ----

    def build_summary(self) -> Dict[str, Any]:
        if self._h5 is None:
            raise RuntimeError("HDF5 file is not open")

        ch_summ: List[Dict[str, Any]] = []
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                try:
                    ch_summ.append({
                        "path": f"/{name}",
                        "name": obj.attrs.get("channel", name.rsplit("/", 1)[-1]),
                        "samples": int(obj.shape[0]),
                        "dtype": str(obj.dtype),
                        "sample_rate": float(obj.attrs.get("sample_rate", self.sample_rate_hz)),
                        "t0": float(obj.attrs.get("t0", 0.0)),
                    })
                except Exception:
                    pass
        self._h5.visititems(_visit)

        try:
            file_size = self.path.stat().st_size
        except Exception:
            file_size = None

        return {
            "path": str(self.path),
            "file_size_bytes": file_size,
            "channels": ch_summ,
            "append_len_samples": self.append_len,
            "compression": f"{self.compression}-{self.gzip_level}",
            "dtype_policy": self.dtype_policy,
        }
    
def write_day_summary(cfg: Config, writer: DayAppendH5Writer, chunks_done: int, log: logging.Logger, elapsed_seconds: float | None = None) -> dict:
    base = writer.build_summary()
    base.update({
        "date": cfg.start_dt.date().isoformat(),
        "start_utc": cfg.start_dt.isoformat(),
        "end_utc": cfg.end_dt.isoformat(),
        "append_interval_s": int(cfg.append_interval),
        "chunks_appended": int(chunks_done),
        "seconds_written": int(chunks_done * cfg.append_interval),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    if elapsed_seconds is not None:
        base["elapsed_seconds"] = round(float(elapsed_seconds), 3)
        # some derived throughput stats (handy in practice)
        total_bytes = base.get("file_size_bytes", None)
        if total_bytes:
            base["throughput_mebibytes_per_s"] = round((total_bytes / (1024**2)) / max(elapsed_seconds, 1e-9), 2)
    
    # move the "channels" at the end for readability
    ch = base.pop("channels", [])
    base["channels"] = ch

    write_json_atomic(cfg.summary_path, base)
    log.info("Summary written: %s", cfg.summary_path)
    return base



def _runs_from_sorted_indices(idxs: List[int]) -> List[List[int]]:
    """Compress a sorted list of integers into runs: [[start, length], ...]."""
    if not idxs:
        return []
    runs: List[List[int]] = []
    start = prev = idxs[0]
    length = 1
    for k in idxs[1:]:
        if k == prev + 1:
            length += 1
        else:
            runs.append([start, length])
            start, length = k, 1
        prev = k
    runs.append([start, length])
    return runs

def write_coverage_json(
    cfg: Config,
    presence: Dict[str, List[int]],
    n_chunks: int,
    channels_all: List[str],
) -> Path:
    """
    Write per-channel 'present_runs' to coverage.json for later NaN-fill conversion.
    'presence' maps channel -> sorted list of chunk indices where data was appended.
    """
    payload: Dict[str, Any] = {
        "date": cfg.start_dt.date().isoformat(),
        "start_utc": cfg.start_dt.isoformat(),
        "end_utc": cfg.end_dt.isoformat(),
        "append_interval_s": int(cfg.append_interval),
        "n_chunks": int(n_chunks),
        "channels": {},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    for ch in channels_all:
        idxs = sorted(presence.get(ch, []))
        runs = _runs_from_sorted_indices(idxs)
        payload["channels"][ch] = {
            "present_runs": runs,
            "total_chunks_present": int(sum(l for _, l in runs)),
        }
    out_path = cfg.day_folder / "coverage.json"
    write_json_atomic(out_path, payload)
    return out_path


# =======================
# Day state (resume) 
# =======================

def load_or_init_day_state(cfg: Config) -> dict:
    """
    Create a minimal day state if missing; otherwise load existing.
    Schema:
      {
        "date": "YYYY-MM-DD",
        "h5_path": ".../trend_YYYYmmdd.h5",
        "append_interval": 100,
        "last_completed_chunk": -1,
        "status": "running|completed|failed",
        "error": null
      }
    """
    if cfg.state_path.is_file():
        with cfg.state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    state = {
        "date": cfg.start_dt.date().isoformat(),
        "h5_path": str(cfg.h5_path),
        "append_interval": int(cfg.append_interval),
        "last_completed_chunk": -1,
        "status": "running",
        "error": None,
    }
    write_json_atomic(cfg.state_path, state)
    return state

def update_day_state(cfg: Config, state: dict, **fields) -> dict:
    state.update(fields)
    write_json_atomic(cfg.state_path, state)
    return state

# ================
# Interrupt flag
# ================
INTERRUPTED = False

def _signal_handler(signum, frame):
    global INTERRUPTED
    INTERRUPTED = True

# ==========================
# gwdama thin reader (v1)
# ==========================

def read_chunk_with_retry(
    t0: datetime,
    t1: datetime,
    channels: list[str],
    ffl_spec: str,
    ffl_path: Path,
    log: logging.Logger,
    retries: int = 1,
) -> tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    """
    Return (data_by_channel, meta_by_channel).
    Skips individual channels that fail; on systemic failures returns ({}, {}).
    Passes GPS floats to gwdama to avoid igwn_segments intersection issues.
    """
    try:
        from gwdama.io import GwDataManager  # lazy import
    except Exception as e:
        log.error("gwdama unavailable: %s", e)
        return {}, {}

    # Convert datetimes to GPS floats (simple fixed-offset helper we already have)
    start_gps = float(utc_to_gps(t0))
    end_gps   = float(utc_to_gps(t1))

    attempt = 0
    while True:
        try:
            dm = GwDataManager()
            result = dm.read_gwdata(
                start=start_gps, end=end_gps, channels=channels,
                ffl_spec=ffl_spec, ffl_path=str(ffl_path),
                return_output=True
            )

            # Normalize output
            if isinstance(result, tuple):
                data_dict = result[0]
                meta_dict = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
            else:
                data_dict = result
                meta_dict = {}

            data_by_ch: Dict[str, np.ndarray] = {}
            meta_by_ch: Dict[str, dict] = {}

            for ch in channels:
                arr = data_dict.get(ch, None)
                if arr is None:
                    continue
                try:
                    arr = np.asarray(arr)
                except Exception:
                    continue
                if arr.ndim != 1:
                    continue  # v1: expect 1-D vectors
                data_by_ch[ch] = arr
                md = meta_dict.get(ch, {})
                meta_by_ch[ch] = {
                    "sample_rate": float(md.get("sample_rate", ASSUMED_TREND_SAMPLE_RATE)),
                    "unit": md.get("unit", None),
                }

            return data_by_ch, meta_by_ch

        except Exception as e:
            attempt += 1
            if attempt > retries:
                log.warning("Read failed (no more retries): %s", e)
                return {}, {}
            log.warning("Read failed (retry %d/%d): %s", attempt, retries, e)



if __name__ == "__main__":
    parser = build_arg_parser()
    ns = parser.parse_args()
    cfg = build_config(ns)

    # Create day folder and initialize logging
    ensure_day_folder(cfg)
    log = setup_logging(cfg)

    # We’ll continue with: load channels → chunk plan → validation…
    # Leaving a breadcrumb for the next step:
    log.info("Day folder ready: %s", cfg.day_folder)

        # === Load channels
    channels = load_channels(cfg.channels_file)
    log.info("Loaded %d unique channels from %s", len(channels), cfg.channels_file)

    # === Build chunk plan
    plan = build_chunk_plan(cfg.start_dt, cfg.end_dt, cfg.append_interval)
    if not plan:
        raise SystemExit("Empty chunk plan (check start/end times and append interval).")
    
    # number of chunks for this day
    n_chunks = len(plan)

    total_secs = (cfg.end_dt - cfg.start_dt).total_seconds()
    covered_secs = n_chunks * cfg.append_interval
    log.info("Chunk plan: %d chunks × %ds = %.0f s (window %.0f s)",
             len(plan), cfg.append_interval, covered_secs, total_secs)
    
     # === Validation (FRCHANNELS + optional probe) → validation.json
    write_validation_json(cfg, channels, log)

    # === Dry run?
    if cfg.dry_run:
        log.info("Dry run requested. Stopping after validation.")
        sys.exit(0)

        # === Signals
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # === State (resume-aware)
    state = load_or_init_day_state(cfg)
    last_completed = int(state.get("last_completed_chunk", -1))
    n_chunks = len(plan)

    # ensure state has planning info
    state.setdefault("plan_start_utc", cfg.start_dt.isoformat())
    state.setdefault("plan_end_utc",   cfg.end_dt.isoformat())
    state.setdefault("plan_n_chunks",  n_chunks)
    update_day_state(cfg, state, last_completed_chunk=last_completed, status="running")

    last_completed = int(state.get("last_completed_chunk", -1))
    prev_n_chunks  = int(state.get("plan_n_chunks", -1))

    # If previous run declared 'completed' but today the plan is longer, we must continue.
    if cfg.resume:
        if state.get("status") == "completed" and last_completed >= n_chunks - 1:
            log.info("Day already completed for current plan: %s", cfg.day_folder)
            sys.exit(0)
        # If plan changed (more chunks), flip status back to running so we append.
        if n_chunks > prev_n_chunks and state.get("status") == "completed":
            update_day_state(cfg, state, status="running")

    # >>> New: compute the first chunk to actually run
    start_idx = 0 if not cfg.resume else (last_completed + 1)
    if start_idx >= n_chunks:
        log.info("Nothing to do: start_idx=%d >= n_chunks=%d", start_idx, n_chunks)
        # keep state as-is; we’re done
        sys.exit(0)

    log.info("Resume: last_completed_chunk=%d, starting_from=%d, total_chunks=%d",
            last_completed, start_idx, n_chunks)

    # === Open HDF5 writer
    append_len = int(cfg.append_interval * ASSUMED_TREND_SAMPLE_RATE)  # 1 Hz
    writer = DayAppendH5Writer(
        path=cfg.h5_path,
        start_utc=cfg.start_dt,
        end_utc=cfg.end_dt,
        append_len=append_len,
        sample_rate_hz=ASSUMED_TREND_SAMPLE_RATE,
        compression=cfg.compression,
        gzip_level=cfg.gzip_level,
        dtype_policy=cfg.dtype_policy,
        log=log,
    )
    writer.open()
    t_job_start = time.perf_counter()
    log.info("Writer ready: chunk append length = %d samples", append_len)

    # === Chunk processing loop (index-driven to avoid off-by-one)
    chunks_done = max(0, last_completed + 1)

    # per-channel presence tracking
    presence: Dict[str, List[int]] = defaultdict(list)  # channel -> list of chunk indices

    try:
        for idx in range(start_idx, n_chunks):
            t0, t1 = plan[idx]

            if INTERRUPTED:
                raise KeyboardInterrupt("interrupted")

            # Read (with GPS conversion inside)
            data_by_ch, meta_by_ch = read_chunk_with_retry(
                t0=t0, t1=t1, channels=channels,
                ffl_spec=cfg.ffl_spec, ffl_path=cfg.ffl_path,
                log=log, retries=1
            )

            if not data_by_ch:
                log.warning("Chunk %d [%s → %s]: no data returned", idx, t0.isoformat(), t1.isoformat())
                # Still advance the resume pointer so we never re-visit this window
                update_day_state(cfg, state, last_completed_chunk=idx, status="running")
                last_completed = idx
                chunks_done = last_completed + 1
                continue

            # Append and flush
            _ = writer.append_chunk(data_by_ch, meta_by_ch, chunk_t0_utc=t0)
            writer.flush()

            # Record presence for channels that actually returned data in this chunk
            for ch in data_by_ch.keys():
                presence[ch].append(idx)

            # Advance resume pointer AFTER successful flush
            update_day_state(cfg, state, last_completed_chunk=idx, status="running")
            last_completed = idx
            chunks_done = last_completed + 1

        # Completed all chunks
        writer.flush()
        elapsed = time.perf_counter() - t_job_start
        cov_path = write_coverage_json(cfg, presence, n_chunks=n_chunks, channels_all=channels)
        log.info("Coverage written: %s", cov_path)
        write_day_summary(cfg, writer, chunks_done=chunks_done, log=log, elapsed_seconds=elapsed)
        update_day_state(cfg, state, status="completed", error=None)
        log.info("Finalize: last_completed_chunk=%d, chunks_done=%d, n_chunks=%d",last_completed, chunks_done, n_chunks)
        log.info("Day completed: %s (chunks=%d)", cfg.day_folder, chunks_done)
        writer.close()
        sys.exit(0)

    except KeyboardInterrupt:
        log.warning("Interrupted by signal; attempting graceful close")
        try:
            writer.flush()
        except Exception:
            pass
        writer.close()
        update_day_state(cfg, state, status="failed", error="interrupted")
        sys.exit(130)  # 128+SIGINT

    except Exception as e:
        log.exception("Fatal error; aborting day: %s", e)
        try:
            writer.flush()
        except Exception:
            pass
        writer.close()
        update_day_state(cfg, state, status="failed", error=str(e))
        sys.exit(2)
    