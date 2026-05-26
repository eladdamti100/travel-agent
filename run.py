"""
Entry point for the AI Travel Planner.

Run:  python run.py
      ./travel.sh
"""

import logging
import warnings

warnings.warn = lambda *args, **kwargs: None  # suppress all third-party warnings
logging.disable(logging.CRITICAL)             # suppress all log output to the terminal

from src.main import run

if __name__ == "__main__":
    run()
