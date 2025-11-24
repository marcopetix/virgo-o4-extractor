import logging
import sys
import time
from utils.configs import ExtractorConfig

def setup_logging_incremental(cfg: ExtractorConfig) -> logging.Logger:
    """
    Set up console + file logging. Timestamps are in UTC.
    """
    logger = logging.getLogger(name="DataExtractor")
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
    logger.info("===== DataExtractor (incremental worker): start =====")
    logger.info(
        "Window: %s → %s | increment-size=%ss | max-retries=%s | resume=%s",
        cfg.start_dt.isoformat(), cfg.end_dt.isoformat(),
        cfg.increment_size, cfg.max_retries, cfg.resume,
    )

    logger.info(
        "Channels: %s | FFL: [%s] %s | Out: %s",
        cfg.channels_file, cfg.ffl_spec, cfg.ffl_path, cfg.day_folder,
    )

    # logger.info(
    #     "Padding missing data: %s | Pruning empty channels: %s",
    #     not cfg.no_padding, not cfg.no_pruning,
    # )

    return logger
