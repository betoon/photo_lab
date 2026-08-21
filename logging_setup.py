"""logging_setup.py — one place to configure PhotoLab's logging.

Call configure_logging() once, as early as possible in main.py (before any
other PhotoLab module that might log is imported, so nothing logs to the
default "no handlers" void first).

Every other module should do:

    import logging
    log = logging.getLogger(__name__)

and use log.debug / log.info / log.warning / log.exception as appropriate.
Nothing else needs to know where the log file lives.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys

from app_paths import log_dir

_LOG_FILENAME = "photolab.log"
_CONFIGURED = False


def configure_logging(level: int = logging.INFO, console: bool = True) -> str:
    """Set up the root logger with a rotating file handler (+ optional console).

    Safe to call more than once; only the first call has an effect. Returns
    the path to the active log file so callers (e.g. a "Send diagnostics"
    dialog, or the ROADMAP's planned exportable problem report) can attach
    or display it.
    """
    global _CONFIGURED
    log_path = os.path.join(log_dir(), _LOG_FILENAME)
    if _CONFIGURED:
        return log_path

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotate at 2MB, keep 5 backups — plenty for a desktop app, never grows
    # unbounded like the hand-copied nppBackup/*.bak files did.
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(fmt)
        root.addHandler(console_handler)

    # Quiet down noisy third-party loggers unless we're debugging.
    logging.getLogger("PIL").setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).info(
        "PhotoLab logging started -> %s", log_path,
    )
    return log_path


def current_log_path() -> str:
    """Path to the active (or about-to-be-active) log file."""
    return os.path.join(log_dir(), _LOG_FILENAME)
