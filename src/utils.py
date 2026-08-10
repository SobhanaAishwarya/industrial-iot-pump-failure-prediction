"""
Small shared utilities: logging setup and JSON I/O helpers.

Kept separate from the domain modules so that logging configuration and
file-persistence conventions live in exactly one place.
"""

import json
import logging
from pathlib import Path
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent, readable format.

    Safe to call repeatedly (e.g. once per module import) without producing
    duplicate log lines, since handlers are only attached once per logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def save_json(data: Any, path: Path) -> None:
    """Write `data` to `path` as pretty-printed JSON, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Read JSON from `path`.

    Raises FileNotFoundError with a clear, actionable message instead of the
    default (often confusing) traceback so calling UI code can surface it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected JSON file not found at '{path}'. "
            "Run the training pipeline first (see src/train.py)."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
