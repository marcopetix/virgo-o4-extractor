from __future__ import annotations

import logging
from datetime import datetime

from utils.configs import (
    build_arg_parser_parallel,
    initialize_config_from_args_parallel,
    initialize_config_from_dict_parallel,
)
from utils.loaders import load_config
from utils.parallel_scheduler import echo_run_header, plan_only, supervise


def setup_parallel_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    # Use UTC in the logs
    logging.Formatter.converter = lambda *args: datetime.utcnow().timetuple()


if __name__ == "__main__":
    parser = build_arg_parser_parallel()
    args = parser.parse_args()

    # Temporary logging level (we may override later based on config)
    setup_parallel_logging(args.log_level)
    log = logging.getLogger("parallel")

    # ParallelConfig from CLI or config file
    if getattr(args, "config_file", None):
        cfg_dict = load_config(args.config_file)
        cfg = initialize_config_from_dict_parallel(cfg_dict)
    else:
        cfg = initialize_config_from_args_parallel(args)

    # Reconfigure logging in case config file changed level (optional)
    setup_parallel_logging(cfg.log_level)
    log = logging.getLogger("parallel")

    echo_run_header(cfg, log)

    if cfg.dry_run:
        plan_only(cfg, log)
    else:
        supervise(cfg, log)
