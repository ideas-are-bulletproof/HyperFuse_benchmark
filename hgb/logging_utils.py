"""
hgb/logging_utils.py
====================

Console + rotating file logger and a tqdm wrapper that degrades gracefully to
a no-op iterator when tqdm is not installed (so the harness still runs in a
minimal environment).  ``epoch_log`` centralises the "log every N epochs"
cadence used by the deep encoders and classifiers.
"""

from __future__ import annotations

import logging
import os
import sys

from . import config as C

_LOGGER = None


def get_logger(name: str = "hgb", logfile: str | None = None) -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None and logfile is None:
        return _LOGGER
    # Don't let a flaky handler (e.g. a network/overlay mount that invalidates a
    # file descriptor mid-run) crash the run or spam fallback tracebacks.
    logging.raiseExceptions = False
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile is not None:
        try:
            os.makedirs(os.path.dirname(logfile), exist_ok=True)
            fh = logging.FileHandler(logfile, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except OSError:
            pass  # console logging still works
    logger.propagate = False
    _LOGGER = logger
    return logger


def tqdm_iter(iterable, desc: str = "", total: int | None = None, leave: bool = False):
    """tqdm progress bar, or a plain iterator if tqdm is unavailable."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True)
    except Exception:
        return iterable


def epoch_log(logger, tag: str, epoch: int, total: int, msg: str,
              every: int = C.LOG_EVERY) -> None:
    """Emit a log line every ``every`` epochs (and always on the final one)."""
    if epoch % every == 0 or epoch == total:
        logger.info(f"    [{tag}] epoch {epoch}/{total} | {msg}")
