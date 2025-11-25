from __future__ import annotations

# =========
# Imports
# =========
import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
import os
from pathlib import Path
from typing import Literal, Optional, Dict, Any, Union

log_level = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
ffl_spec = Literal['V1raw', 'V1trend', 'V1trend100']

FFL_TO_SPEC_MAP = {
    'V1raw': Path("/virgoData/ffl/raw.ffl"),
    'V1trend': Path("/virgoData/ffl/trend.ffl"),
    'V1trend100': Path("/virgoData/ffl/trend100s.ffl")
}

@dataclass()
class ExtractorConfig:
    # Core time window (UTC)
    start_dt: Optional[datetime]
    end_dt: Optional[datetime]

    # Core time window (GPS)
    start_gps: Optional[int]
    end_gps: Optional[int]

    # Duration
    duration: Optional[timedelta | int]

    # I/O inputs
    channels_file: Optional[Path]                                         # either a .txt, .json or .csv file listing channels to extract
    ffl_spec: Optional[ffl_spec | str]                          # Pre-encoded specs of the ffl file to read (mainly V1raw, V1trend) (string to allow for extensions)
    ffl_path: Optional[Path]                                    # Alternative to 'ffl_spec' (if the path to a local ffl file is available, 'ffl_spec' is ignored) 

    # Behavior flags
    resume: bool = False
    log_level: log_level = 'INFO'                               # 'INFO' | 'DEBUG' 

    # Outputs (resolved paths)
    out_root: Optional[Path] = None                                              # base output directory
    day_folder: Optional[Path] = None                                            # out_root/{ffl_spec}-YYYY-MM-DD/
    dataset_path: Optional[Path] = None                                          # day_folder/{ffl_spec}_YYYYmmdd.h5
    state_path: Optional[Path] = None                                            # day_folder/state_day.json
    summary_path: Optional[Path] = None                                          # day_folder/{ffl_spec}_YYYYmmdd.summary.json
    log_path: Optional[Path] = None                                              # day_folder/worker.log

    # Extraction parameters
    increment_size: int = 600                                   # duration of each read chunk in seconds
    max_retries: int = 3                                        # number of retries for reading data
    compression: Optional[str] = "lzf"                          # compression algorithm for HDF5 (e.g., 'gzip', 'lzf', None)
    produce_summary: bool = True                                # whether to produce a summary JSON file after extraction  

# =========
# Config initialization
# =========
def initialize_config_from_args_incremental(args: argparse.Namespace) -> ExtractorConfig:
    """
    Initialize ExtractorConfig from argparse.Namespace.
    Perform input validation and build output paths.
    """
    # Basic config initialization
    config = ExtractorConfig(
        start_dt=args.start_dt,
        end_dt=args.end_dt,
        start_gps=args.start_gps,
        end_gps=args.end_gps,
        duration=args.duration,
        channels_file=Path(args.channels_file),
        ffl_spec=args.ffl_spec,
        ffl_path=Path(args.ffl_path) if args.ffl_path else None,
        resume=args.resume,
        log_level=args.log_level,
        out_root=Path(args.out_root),
        day_folder=None,          # to be set later
        dataset_path=None,       # to be set later
        state_path=None,         # to be set later
        summary_path=None,       # to be set later
        log_path=None,           # to be set later
        increment_size=args.increment_size,
        max_retries=args.max_retries,
    )
    
    assert config.increment_size > 0, "increment_size must be positive."
    assert config.max_retries >= 0, "max_retries must be non-negative."

    _validate_start_end_duration(config)
    _validate_channels(config)
    _validate_ffl(config)

    _build_output_paths(config)
    _ensure_day_folder(config)

    return config

def initialize_config_from_dict_incremental(cfg_dict: Dict[str, Any]) -> ExtractorConfig:
    """
    Initialize ExtractorConfig from a dictionary.
    Perform input validation and build output paths.
    """
    # Basic config initialization
    config = ExtractorConfig(
        start_dt=cfg_dict.get('start_dt'),
        end_dt=cfg_dict.get('end_dt'),
        start_gps=cfg_dict.get('start_gps'),
        end_gps=cfg_dict.get('end_gps'),
        duration=cfg_dict.get('duration'),
        channels_file=Path(cfg_dict['channels_file']),
        ffl_spec=cfg_dict.get('ffl_spec'),
        ffl_path=Path(cfg_dict['ffl_path']) if 'ffl_path' in cfg_dict else None,
        resume=cfg_dict.get('resume', False),
        log_level=cfg_dict.get('log_level', 'INFO'),
        out_root=Path(cfg_dict['out_root']) or Path(cfg_dict.get('out_dir')),
        day_folder=None,          # to be set later
        dataset_path=None,       # to be set later
        state_path=None,         # to be set later
        summary_path=None,       # to be set later
        log_path=None,           # to be set later
        increment_size=cfg_dict.get('increment_size', 600),
        max_retries=cfg_dict.get('max_retries', 3),
    )
    
    assert config.increment_size > 0, "increment_size must be positive."
    assert config.max_retries >= 0, "max_retries must be non-negative."

    _validate_start_end_duration(config)
    _validate_channels(config)
    _validate_ffl(config)
        
    _build_output_paths(config)
    _ensure_day_folder(config)

    return config

# =========
# Config validation
# =========
def _validate_start_end_duration(config: ExtractorConfig) -> None:
    """
    Validate the ExtractorConfig parameters (start_dt, end_dt, start_gps, end_gps duration).
    Raise ValueError if any parameter is invalid.
    """

    # --- start, end, duration validation and initialization --- (most of these are redundant, for now we do them all here)

    # ensure either start_dt/end_dt or start_gps/end_gps or duration is provided
    assert (config.start_dt is not None and config.end_dt is not None) or (config.start_gps is not None and config.end_gps is not None) or \
       (config.start_dt is not None and config.duration is not None) or (config.end_dt is not None and config.duration is not None) or \
       (config.start_gps is not None and config.duration is not None) or (config.end_gps is not None and config.duration is not None), "At least one of these pairs must be provided: (start_dt, end_dt), (start_gps, end_gps), (start_dt, duration), (end_dt, duration), (start_gps, duration), (end_gps, duration)."
        
    # Now we convert and fill in missing values as needed
    config.start_dt = _parse_utc(config.start_dt) or ( _gps_to_utc(config.start_gps) if config.start_gps is not None else None) or \
                      ( _parse_utc(config.end_dt) - timedelta(seconds=config.duration) if config.end_dt is not None and config.duration is not None else None)
    config.end_dt = _parse_utc(config.end_dt) or ( _gps_to_utc(config.end_gps) if config.end_gps is not None else None) or \
                    ( config.start_dt + timedelta(seconds=config.duration) if config.start_dt is not None and config.duration is not None else None)
    config.start_gps = config.start_gps or ( _utc_to_gps(config.start_dt) if config.start_dt is not None else None) or \
                       ( config.end_gps - config.duration if config.end_gps is not None and config.duration is not None else None)
    config.end_gps = config.end_gps or ( _utc_to_gps(config.end_dt) if config.end_dt is not None else None) or \
                     ( config.start_gps + config.duration if config.start_gps is not None and config.duration is not None else None)
    config.duration = config.duration or (config.end_gps - config.start_gps if config.start_gps is not None and config.end_gps is not None else None) or \
                        (int((config.end_dt - config.start_dt).total_seconds()) if config.start_dt is not None and config.end_dt is not None else None)

    # Now we check for consistency and validity of start/end/duration
    now_utc = datetime.now(timezone.utc)
    if config.start_dt >= config.end_dt or config.start_gps >= config.end_gps:
            raise ValueError("start_dt/start_gps must be earlier than end_dt/end_gps.")
    if config.end_dt > now_utc or config.end_gps > _utc_to_gps(now_utc):
        raise ValueError("end_dt/end_gps cannot be in the future.")    
    
    # Ensure start_dt and end_dt are timezone-aware UTC datetimes
    if config.start_dt.tzinfo is None or config.start_dt.tzinfo.utcoffset(config.start_dt) != timedelta(0):
        raise ValueError("start_dt must be a timezone-aware UTC datetime.")
    if config.end_dt.tzinfo is None or config.end_dt.tzinfo.utcoffset(config.end_dt) != timedelta(0):
        raise ValueError("end_dt must be a timezone-aware UTC datetime.")

    # Ensure start_gps and end_gps are integers
    if not isinstance(config.start_gps, int):
        raise ValueError("start_gps must be an integer.")
    if not isinstance(config.end_gps, int):
        raise ValueError("end_gps must be an integer.")
    
    # Ensure duration is consistent with start and end and is positive
    assert config.duration is not None, "duration must be set at this point."
    assert config.duration == (config.end_gps - config.start_gps), "duration must equal end_gps - start_gps."
    assert config.duration == int((config.end_dt - config.start_dt).total_seconds()), "duration must equal end_dt - start_dt in seconds."
    assert config.duration > 0, "duration must be positive."

def _validate_channels(config: ExtractorConfig) -> None:
    """
    Validate the channels file specified in the config.
    Raise ValueError if the file does not exist or is not readable.
    """
    # if channels_file does not exist or is not readable, raise error
    if not config.channels_file.exists():
        raise ValueError(f"Channels file '{config.channels_file}' does not exist.")
    if not config.channels_file.is_file():
        raise ValueError(f"Channels file '{config.channels_file}' is not a file.")
    if not os.access(config.channels_file, os.R_OK):
        raise ValueError(f"Channels file '{config.channels_file}' is not readable.")
    
    # if channels_file is txt, json or csv
    if config.channels_file.suffix not in ['.txt', '.json', '.csv']:
        raise ValueError(f"Channels file '{config.channels_file}' must be a .txt, .json, or .csv file.")
    
    # if channels_file is empty, raise error
    if config.channels_file.stat().st_size == 0:
        raise ValueError(f"Channels file '{config.channels_file}' is empty.")
    
def _validate_ffl(config: ExtractorConfig) -> None:
    """
    Validate the ffl_spec and ffl_path in the config.
    Raise ValueError if any parameter is invalid.
    """
    
    assert config.ffl_spec is not None or config.ffl_path is not None, "Either ffl_spec or ffl_path must be provided."

    # if ffl_path is provided, ensure it exists and is readable
    if config.ffl_path is not None:
        if not config.ffl_path.exists():
            raise ValueError(f"FFL file '{config.ffl_path}' does not exist.")
        if not config.ffl_path.is_file():
            raise ValueError(f"FFL file '{config.ffl_path}' is not a file.")
        if not os.access(config.ffl_path, os.R_OK):
            raise ValueError(f"FFL file '{config.ffl_path}' is not readable.")
    
    # if ffl_spec is provided, ensure it is one of the allowed literals
    if config.ffl_spec is not None:
        if config.ffl_spec not in FFL_TO_SPEC_MAP.keys():
            raise ValueError(f"ffl_spec '{config.ffl_spec}' is not valid. Allowed values are: {list(FFL_TO_SPEC_MAP.keys())}.")
    
    # if both ffl_spec and ffl_path are provided, ffl_path takes precedence but let's ensure they are consistent
    if config.ffl_path is not None and config.ffl_spec is not None:
        expected_path = FFL_TO_SPEC_MAP[config.ffl_spec]
        if config.ffl_path != expected_path:
            raise ValueError(f"ffl_path '{config.ffl_path}' does not match the expected path for ffl_spec '{config.ffl_spec}': '{expected_path}'.")
    
    # if only ffl_spec is provided, set ffl_path accordingly
    elif config.ffl_spec is not None:
        object.__setattr__(config, 'ffl_path', FFL_TO_SPEC_MAP[config.ffl_spec])
    elif config.ffl_path is not None:
        # if only ffl_path is provided, try to infer ffl_spec from filename
        filename = config.ffl_path.name.lower()
        inferred_spec = None
        for spec, path in FFL_TO_SPEC_MAP.items():
            if path.name.lower() == filename:
                inferred_spec = spec
                break
        if inferred_spec is None:
            raise ValueError(f"Could not infer ffl_spec from ffl_path '{config.ffl_path}'. Please provide a valid ffl_spec.")
        object.__setattr__(config, 'ffl_spec', inferred_spec)

# =========
# Datetime utils
# =========
GPS_EPOCH_UTC = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)  # GPS epoch in UTC
GPS_LEAP_SECONDS = 18                                               # GPS-UTC offset has been 18s since 2017-01-01. This is a fixed offset for our purposes.

def _utc_to_gps(dt: datetime) -> int: 
    """Convert aware UTC datetime → GPS seconds (simple, fixed 18s offset)."""
    if dt.tzinfo is None:
        raise ValueError("utc_to_gps requires an aware UTC datetime")
    return int((dt - GPS_EPOCH_UTC).total_seconds() + GPS_LEAP_SECONDS)

def _gps_to_utc(gps: float) -> datetime:
    """Convert GPS seconds → aware UTC datetime (simple, fixed 18s offset)."""
    return GPS_EPOCH_UTC + timedelta(seconds=(gps - GPS_LEAP_SECONDS))

def _parse_utc(ts: Optional[Union[str, datetime]]) -> Optional[datetime]:
    """
    Parse 'YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD' or 'DD-MM-YYYY' to aware UTC datetime.
    If a datetime is passed, normalize to UTC and return it.
    If None is passed, return None.
    """
    if ts is None:
        return None
    if isinstance(ts, datetime):
        # normalize to UTC
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    # from here on, ts is a string
    formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d-%m-%Y']
    for fmt in formats:
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(
        f"Timestamp '{ts}' is not in a recognized format. "
        f"Expected formats: {formats}"
    )

def _str_date_folder(dt: datetime) -> str:
    return dt.date().isoformat()  # 'YYYY-MM-DD'

# =========
# Directory utils
# =========
def _build_output_paths(config: ExtractorConfig) -> None:
    """
    Given an ExtractorConfig, build the output directory and file paths.
    Updates the config in place to add day_folder, dataset_path, state_path, summary_path, log_path.
    """
    if config.start_dt is None:
        raise ValueError("start_dt must be set to build output paths.")

    day_folder = config.out_root / f"{config.ffl_spec}-{_str_date_folder(config.start_dt)}"
    dataset_path = day_folder / f"{config.ffl_spec}_{config.start_dt.strftime('%Y%m%d')}.h5"
    state_path = day_folder / "state_day.json"
    summary_path = day_folder / f"{config.ffl_spec}_{config.start_dt.strftime('%Y%m%d')}.summary.json"
    log_path = day_folder / "worker.log"

    object.__setattr__(config, 'day_folder', day_folder)
    object.__setattr__(config, 'dataset_path', dataset_path)
    object.__setattr__(config, 'state_path', state_path)
    object.__setattr__(config, 'summary_path', summary_path)
    object.__setattr__(config, 'log_path', log_path)

def _ensure_day_folder(cfg: ExtractorConfig) -> None:
    """
    Create the day folder (OUT_ROOT/YYYY-MM-DD) if it doesn't exist.
    This function does NOT create the HDF5 file itself.
    """
    cfg.day_folder.mkdir(parents=True, exist_ok=True)

# =========
# Argument parsing utils
# =========
def build_arg_parser_incremental() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="incremental_conversion.py",
        description=(
            "Convert Virgo GWF → daily HDF5 using a Incremental Writing strategy "
            "(single-day worker)."
            "Reads data in chunks and appends to HDF5 file."
            "Can be initialized via config file or command-line args."
            "The time window to extract must be specified via start/end times or duration (either start + end or start/end + duration)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    p.add_argument("--config-file", "-c", type=Path, 
                   help="Path to a toml config file with all parameters.")
    
    # Required time window
    p.add_argument("--start-dt", type=_parse_utc,
                   help=f"Start timestamp in UTC. Examples: '2024-01-15 12:34:56' or '2024-01-15'.")
    p.add_argument("--end-dt", type=_parse_utc,
                   help=f"End timestamp in UTC. Examples: '2024-01-15 12:34:56' or '2024-01-15'.")
    p.add_argument("--start-gps", type=int,
                   help="Start time in GPS seconds.")
    p.add_argument("--end-gps", type=int,
                   help="End time in GPS seconds.")
    p.add_argument("--duration", type=int,
                   help="Duration in seconds. Examples: 86400 for one day. 604800 for one week.")

    # Required I/O inputs
    p.add_argument("--channels-file", type=Path,
                   help="Path to a .txt/.json/.csv file with one channel per line.")
    p.add_argument("--ffl-path", type=Path,
                   help="Path to the FFL file (e.g., /virgoData/ffl/trend.ffl).")
    p.add_argument("--ffl-spec", type=str,
                   help="FFL spec name (e.g., V1trend).")

    # Output root
    p.add_argument("--out", dest="out_root", type=Path,
                   help="Root output directory where the day folder will be created.")

    # Behavior
    p.add_argument("--increment-size", type=int, default=600,
                   help="Size of each data read increment in seconds.")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Maximum number of retries on failure.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from state_day.json if present.")
    p.add_argument("--log-level", choices=["INFO", "DEBUG"], default="INFO",
                   help="Logging verbosity.")

    return p


# =========
# Parallel launcher config
# =========

def _parse_date(s: str) -> date:
    """Parse YYYY-MM-DD into a date object (for parallel launcher)."""
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date '{s}' (expected YYYY-MM-DD)") from e


def _positive_int(name: str, v: str) -> int:
    """Helper for positive integer CLI args (parallel launcher)."""
    try:
        i = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--{name} must be an integer")
    if i <= 0:
        raise argparse.ArgumentTypeError(f"--{name} must be > 0")
    return i

@dataclass()
class ParallelConfig:
    # Required
    start_date: date
    end_date: date
    channels_file: Path
    ffl_path: Path
    ffl_spec: str
    out_dir: Path

    # Limits / smoke tests
    limit_days: int = 0              # 0 = no limit
    minutes_per_day: int = 0         # 0 = full day

    # Optional / tuning
    increment_size: int = 100       # seconds (passed as increment-size to worker)
    concurrency: int = 8             # max concurrent worker processes
    log_level: str = "INFO"
    resume: bool = True
    max_retries: int = 2
    stagger_seconds: int = 5         # seconds between spawns
    mem_guard_mb: int = 0            # 0 = disabled
    dry_run: bool = False
    compression: str = "gzip"        # currently not forwarded to worker CLI

    # Worker entrypoint (override for tests / custom layout)
    worker_path: Path = Path("scripts/incremental_conversion.py")

    # Derived (filled by initializer)
    run_utc_started: str = ""
    args_echo: str = ""

def build_arg_parser_parallel() -> argparse.ArgumentParser:
    """
    Argument parser for the parallel launcher.
    """
    p = argparse.ArgumentParser(
        prog="parallel_conversion.py",
        description="Parallel launcher for daily GWF→HDF5 conversion (one subprocess per day).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Optional config file for the launcher itself
    p.add_argument(
        "--config-file", "-c", type=Path,
        help="Path to a TOML/JSON config file for the parallel launcher."
    )

    # Required (if no config-file is used)
    p.add_argument(
        "--start-date", type=_parse_date,
        help="Inclusive start date (YYYY-MM-DD)."
    )
    p.add_argument(
        "--end-date", type=_parse_date,
        help="Inclusive end date (YYYY-MM-DD)."
    )
    p.add_argument(
        "--channels-file", type=Path,
        help="TXT file with channel names (one per line)."
    )
    p.add_argument(
        "--ffl-path", type=Path,
        help="Frame file list (FFL)."
    )
    p.add_argument(
        "--ffl-spec", type=str,
        help="FFL spec name (e.g. V1trend)."
    )
    p.add_argument(
        "--out", dest="out_dir", type=Path,
        help="Output root directory."
    )

    # Optional / tuning
    p.add_argument(
        "--increment-size",
        type=lambda v: _positive_int("increment-size", v),
        default=100,
        help="Append interval in seconds passed to the worker (increment-size)."
    )
    p.add_argument(
        "--concurrency",
        type=lambda v: _positive_int("concurrency", v),
        default=8,
        help="Max concurrent worker processes."
    )
    p.add_argument(
        "--max-retries",
        type=lambda v: _positive_int("max-retries", v),
        default=2,
        help="Max retries per day on nonzero worker exit."
    )
    p.add_argument(
        "--stagger-seconds",
        type=lambda v: _positive_int("stagger-seconds", v),
        default=5,
        help="Delay between spawns to avoid NFS bursts."
    )
    p.add_argument(
        "--mem-guard-mb", type=int, default=0,
        help="If >0, do not spawn new jobs when MemAvailable < this many MB."
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default="INFO",
        help="Parallel launcher logging verbosity."
    )
    p.add_argument(
        "--resume", action="store_true", default=False,
        help="Pass --resume to workers and skip fully completed days."
    )
    p.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Plan only; do not spawn workers."
    )
    p.add_argument(
        "--compression",
        type=str,
        choices=["gzip", "lzf", "none"],
        default="gzip",
        help="Compression flag to record in the parallel config (not yet used by worker CLI)."
    )

    # Testing / override
    p.add_argument(
        "--worker-path", type=Path,
        default=Path("scripts/incremental_conversion.py"),
        help="Path to the single-day worker script."
    )
    p.add_argument(
        "--limit-days", type=int, default=0,
        help="Limit how many days to schedule (0 = no limit)."
    )
    p.add_argument(
        "--minutes-per-day", type=int, default=0,
        help="Extract only the first M minutes of each day (0 = full day)."
    )

    return p

def initialize_config_from_args_parallel(ns: argparse.Namespace) -> ParallelConfig:
    """
    Initialize ParallelConfig from argparse.Namespace (CLI args).
    """
    # If using a config file, this function should not be called
    if ns.config_file is not None:
        raise ValueError("initialize_config_from_args_parallel called with config_file set")

    # Validate date range
    if ns.start_date is None or ns.end_date is None:
        raise SystemExit("Both --start-date and --end-date are required (or use --config-file).")
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
        increment_size=int(ns.increment_size),
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
        limit_days=int(ns.limit_days),
        minutes_per_day=int(ns.minutes_per_day),
        compression=str(ns.compression),
    )

    # Validate limits
    if cfg.limit_days < 0:
        raise SystemExit("--limit-days must be >= 0")
    if cfg.minutes_per_day < 0:
        raise SystemExit("--minutes-per-day must be >= 0")
    if cfg.minutes_per_day and cfg.minutes_per_day < cfg.increment_size // 60:
        # keep it simple: we want at least one full chunk
        raise SystemExit("--minutes-per-day must be >= increment_size/60")

    return cfg

def initialize_config_from_dict_parallel(cfg_dict: Dict[str, Any]) -> ParallelConfig:
    """
    Initialize ParallelConfig from a dictionary (e.g. loaded from TOML/JSON).
    """
    # Required fields
    try:
        start_date = _parse_date(cfg_dict["start_date"])
        end_date = _parse_date(cfg_dict["end_date"])
    except KeyError as e:
        raise SystemExit(f"Missing required key in parallel config: {e.args[0]}") from e

    channels_file = Path(cfg_dict["channels_file"])
    ffl_path = Path(cfg_dict["ffl_path"])
    ffl_spec = str(cfg_dict.get("ffl_spec", "V1trend"))
    out_dir = Path(cfg_dict["out_dir"]) or Path(cfg_dict["out_root"])

    # Validate basic things
    if start_date > end_date:
        raise SystemExit("start_date must be <= end_date in parallel config")

    if not channels_file.is_file():
        raise SystemExit(f"channels file not found: {channels_file}")
    if not ffl_path.is_file():
        raise SystemExit(f"ffl file not found: {ffl_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    worker_path = Path(cfg_dict.get("worker_path", "scripts/incremental_conversion.py"))
    if not worker_path.exists():
        raise SystemExit(f"worker script not found: {worker_path}")

    # Optional/tuning with defaults
    increment_size = int(cfg_dict.get("increment_size", 100))
    concurrency = int(cfg_dict.get("concurrency", 8))
    log_level_str = str(cfg_dict.get("log_level", "INFO"))
    resume = bool(cfg_dict.get("resume", False))
    max_retries = int(cfg_dict.get("max_retries", 2))
    stagger_seconds = int(cfg_dict.get("stagger_seconds", 5))
    mem_guard_mb = int(cfg_dict.get("mem_guard_mb", 0))
    dry_run = bool(cfg_dict.get("dry_run", False))
    limit_days = int(cfg_dict.get("limit_days", 0))
    minutes_per_day = int(cfg_dict.get("minutes_per_day", 0))
    compression = str(cfg_dict.get("compression", "gzip"))

    cfg = ParallelConfig(
        start_date=start_date,
        end_date=end_date,
        channels_file=channels_file.resolve(),
        ffl_path=ffl_path.resolve(),
        ffl_spec=ffl_spec,
        out_dir=out_dir.resolve(),
        increment_size=increment_size,
        concurrency=concurrency,
        log_level=log_level_str,
        resume=resume,
        max_retries=max_retries,
        stagger_seconds=stagger_seconds,
        mem_guard_mb=mem_guard_mb,
        dry_run=dry_run,
        worker_path=worker_path.resolve(),
        run_utc_started=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        args_echo="(from parallel config file)",
        limit_days=limit_days,
        minutes_per_day=minutes_per_day,
        compression=compression,
    )

    # Validate limits
    if cfg.limit_days < 0:
        raise SystemExit("parallel config: limit_days must be >= 0")
    if cfg.minutes_per_day < 0:
        raise SystemExit("parallel config: minutes_per_day must be >= 0")
    if cfg.minutes_per_day and cfg.minutes_per_day < cfg.increment_size // 60:
        raise SystemExit("parallel config: minutes_per_day must be >= increment_size/60")

    return cfg
