"""
Logging factory — returns a consistently formatted named logger.
"""

import logging
import sys
from pathlib import Path


LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "travel_agent.log"


_CACHE_LOGGERS = {"cache_checker", "semantic_cache", "cache_store"}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with console and file handlers.

    Console output is suppressed for non-cache loggers so the terminal only
    shows cache hit/miss status. All messages still go to the log file.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
    has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    if not has_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        # Only cache-related loggers print to console
        console_handler.setLevel(logging.INFO if name in _CACHE_LOGGERS else logging.WARNING)
        logger.addHandler(console_handler)

    if not has_file:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger