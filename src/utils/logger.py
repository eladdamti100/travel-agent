import logging
import sys


def get_logger(name: str, level: int = logging.WARNING) -> logging.Logger:
    """
    Return a named logger with a consistent format.
    Avoids adding duplicate handlers if called multiple times with the same name.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
