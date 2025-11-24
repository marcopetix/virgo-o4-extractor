"""
Single-day incremental Virgo GWF -> HDF5 converter.

- Parses CLI or config file into an ExtractorConfig
- Sets up logging
- Loads the channel list
- Runs the IncrementalDataExtractor worker
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import signal
from typing import Optional

from utils.configs import (
    build_arg_parser_incremental,
    initialize_config_from_args_incremental,
    initialize_config_from_dict_incremental,
    ExtractorConfig,
)
from utils.loggers import setup_logging_incremental
from utils.loaders import load_channels, load_config
from classes.incremental_worker import IncrementalDataExtractor


# ================
# Signal handling
# ================

def _signal_handler(signum, frame):
    """
    Convert SIGINT/SIGTERM into a KeyboardInterrupt so that the main
    try/except block can handle a graceful shutdown.
    """
    raise KeyboardInterrupt


# ================
# Main worker
# ================

def main(config: ExtractorConfig) -> None:
    # Logging setup
    logger = setup_logging_incremental(config)
    logger.debug("Configuration: %r", config)

    # Channels loading
    channels = load_channels(config.channels_file)
    logger.info("Loaded %d channels from %s", len(channels), config.channels_file)

    # Prepare to handle interruptions (termination signals)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Prepare writer
    writer = IncrementalDataExtractor(config, channels, logger)
    writer.open()
    logger.debug("Opened HDF5 file at %s", writer.dataset_path)

    try:
        # Main extraction loop
        writer.run_extraction()

        # Mark state as completed (status only; last_completed_chunk already updated)
        writer.update_day_state(status="completed", error=None)

        writer.close()

    except KeyboardInterrupt:
        logger.warning("Interrupted by signal; attempting graceful close")
        try:
            writer.flush()
        except Exception:
            pass
        writer.close()
        writer.update_day_state(status="failed", error="interrupted")
        sys.exit(130)  # 128 + SIGINT

    except Exception as e:
        logger.exception("Fatal error; aborting day: %s", e)
        try:
            writer.flush()
        except Exception:
            pass
        writer.close()
        writer.update_day_state(status="failed", error=str(e))
        sys.exit(2)  # general error


# ================
# CLI entry point
# ================

if __name__ == "__main__":
    # Arg parsing / Config file reading
    parser = build_arg_parser_incremental()
    args = parser.parse_args()

    # Initialize config
    if args.config_file:
        config_dict = load_config(args.config_file)
        cfg = initialize_config_from_dict_incremental(config_dict)
    else:
        cfg = initialize_config_from_args_incremental(args)

    # Run main function
    main(cfg)
