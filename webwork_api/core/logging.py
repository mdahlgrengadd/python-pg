"""
Structured logging configuration.

Provides consistent, structured logging across the application.
"""

import sys
import logging
from typing import Any, Dict
from datetime import datetime
import json
from pathlib import Path

from .config import settings


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Human-readable text log formatter"""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def setup_logging() -> None:
    """Configure application logging"""

    # Determine log level
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Create formatter
    if settings.LOG_FORMAT == "json":
        formatter = StructuredFormatter()
    else:
        formatter = TextFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    handlers = [console_handler]

    # File handler (if configured)
    if settings.LOG_FILE:
        log_path = Path(settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )

    # Silence noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Enhanced logger with structured context"""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add context to log messages"""
        # Extract extra data
        extra_data = kwargs.pop("extra_data", {})

        # Add to record
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        kwargs["extra"]["extra_data"] = {
            **self.extra,
            **extra_data
        }

        return msg, kwargs


def get_context_logger(name: str, **context) -> LoggerAdapter:
    """Get logger with permanent context"""
    logger = get_logger(name)
    return LoggerAdapter(logger, context)


# Example usage:
# logger = get_logger(__name__)
# logger.info("Problem loaded", extra_data={"problem_id": "test", "seed": 12345})
#
# Or with permanent context:
# logger = get_context_logger(__name__, user_id="123", session_id="abc")
# logger.info("Answer submitted")  # Automatically includes user_id and session_id
