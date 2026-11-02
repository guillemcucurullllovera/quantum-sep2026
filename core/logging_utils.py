"""Structured logging setup, shared so every module's logs are consistent
and machine-parseable when needed (experiment runs, hardware job lifecycle)."""
from __future__ import annotations

import json
import logging
import sys
import time

from core.config import get_settings


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str, *, json_output: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JSONFormatter() if json_output else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, get_settings().log_level.upper(), logging.INFO))
    logger.propagate = False
    return logger
