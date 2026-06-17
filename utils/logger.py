"""
VisionAI Structured Logging Module
───────────────────────────────────
Replaces all print() calls with a proper logging system.
Supports console + file output with timestamps and log levels.
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime

from utils.debug_log import DEBUG_LOG_PATH


_LOGGER_NAME = "visionai"
_logger = None


def get_logger(log_dir: str = None, level: str = "INFO") -> logging.Logger:
    """
    Returns the singleton VisionAI logger.
    First call initializes console + optional file handler.
    Subsequent calls return the existing logger.
    """
    global _logger
    if _logger is not None:
        return _logger

    _logger = logging.getLogger(_LOGGER_NAME)
    _logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    _logger.propagate = False

    
    console_fmt = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_fmt)
    _logger.addHandler(console_handler)

    debug_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    debug_handler = logging.FileHandler(str(DEBUG_LOG_PATH), mode="a", encoding="utf-8")
    debug_handler.setFormatter(debug_fmt)
    _logger.addHandler(debug_handler)
    _logger.info(f"Append debug log: {DEBUG_LOG_PATH}")

    
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = log_path / f"pipeline_{timestamp}.log"

        file_fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler.setFormatter(file_fmt)
        _logger.addHandler(file_handler)
        _logger.info(f"Log file: {file_path}")

    return _logger


_run_handlers = []


def attach_run_logfile(log_dir: str, level: str = "INFO") -> str:
    """Attach a per-run file handler to the singleton logger; return the log file path.

    The singleton is created at import time (before any run dir exists), so its ``log_dir``
    branch never fires for the real run. This adds the file handler after the run dir is
    created. Handlers from previous runs are detached first, so a long-lived process (the
    API server) doesn't bleed one run's logs into the next or leak file descriptors.
    """
    global _run_handlers
    logger = get_logger(level=level)

    for hnd in _run_handlers:
        try:
            hnd.close()
            logger.removeHandler(hnd)
        except Exception:
            pass
    _run_handlers = []

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = log_path / f"pipeline_{timestamp}.log"

    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    _run_handlers.append(file_handler)
    logger.info(f"Per-run log file: {file_path}")
    return str(file_path)


def reset_logger():
    """Reset the logger (useful for tests or re-initialization)."""
    global _logger
    if _logger:
        for handler in _logger.handlers[:]:
            handler.close()
            _logger.removeHandler(handler)
    _logger = None
