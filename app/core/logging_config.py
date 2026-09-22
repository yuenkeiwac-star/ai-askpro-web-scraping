"""Central logging setup.

Two distinct log line styles are produced, both written to the same file
(and echoed to the console), through one shared, size-rotated handler:

- General application logs, emitted via ``logging.getLogger(__name__)`` in
  any module:
      2026-05-19 07:41:57,099 - INFO - scraper.py - <message>

- API request lifecycle logs, emitted by the middleware in app/main.py via
  ``logging.getLogger(API_LOGGER_NAME)``:
      2026-05-19 07:57:04,476 | INFO | app | [pid] START POST /scrape
      2026-05-19 07:57:05,821 | INFO | app | [pid] END status=200 duration=1.35s

A single handler (not one per logger) is deliberate: two independent
RotatingFileHandlers pointed at the same path would each track their own
size/rollover state and step on each other the moment either one rotated.
Routing everything through one handler and picking the output format inside
a custom Formatter avoids that.

Call setup_logging() once, before anything else logs (app/main.py does this
at import time).
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_FILE = LOG_DIR / "askpro_web_scraping.log"
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", 100 * 1024 * 1024))  # 100MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 10))

GENERAL_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s - %(message)s"
# "app" is hardcoded (not %(name)s) deliberately: this project's package is
# itself named "app", so every module logger (app.api.scrape, etc.) is a
# descendant of a logger literally named "app" -- using that name here would
# make general logs propagate into this handler instead of the root one.
API_FORMAT = "%(asctime)s | %(levelname)s | app | %(message)s"
API_LOGGER_NAME = "api_access"


class _DualStyleFormatter(logging.Formatter):
    """Applies API_FORMAT to API_LOGGER_NAME records, GENERAL_FORMAT to all others."""

    def __init__(self) -> None:
        super().__init__()
        self._general = logging.Formatter(GENERAL_FORMAT)
        self._api = logging.Formatter(API_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        if record.name == API_LOGGER_NAME or record.name.startswith(f"{API_LOGGER_NAME}."):
            return self._api.format(record)
        return self._general.format(record)


def setup_logging(level: int = logging.INFO) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = _DualStyleFormatter()

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # No handlers of its own -- propagates up to the root logger's shared
    # handler above, which picks API_FORMAT for it via _DualStyleFormatter.
    api_logger = logging.getLogger(API_LOGGER_NAME)
    api_logger.setLevel(level)
    api_logger.handlers.clear()
    api_logger.propagate = True
