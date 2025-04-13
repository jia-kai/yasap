from libyasap.utils import CustomFormatter

from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional, ClassVar
import logging

class RollingBufferHandler(logging.Handler):
    """A logging handler that maintains a rolling buffer of log messages."""

    _lock: Lock

    def __init__(self, buffer_size=1000):
        super().__init__()
        self.buffer = deque(maxlen=buffer_size)
        self._lock = Lock()

    def emit(self, record):
        """Add a record to the buffer."""
        with self._lock:
            formatted_record = self.format(record)
            self.buffer.append(formatted_record)

    def get_buffer(self):
        """Get the current buffer contents."""
        with self._lock:
            return list(self.buffer)


class LogManager:
    """Manages logging configuration and provides access to log buffer."""

    _instance: ClassVar[Optional['LogManager']] = None

    @classmethod
    def get_instance(cls) -> 'LogManager':
        """Get the singleton instance of LogManager."""
        if cls._instance is None:
            cls._instance = LogManager()
        return cls._instance

    def __init__(self):
        """Initialize the log manager."""
        if self._instance is not None:
            raise RuntimeError("LogManager is a singleton")

        LogManager._instance = self

        # Create logs directory if it doesn't exist
        self.logs_dir = Path(__file__).parent.parent / 'logs'
        self.logs_dir.mkdir(exist_ok=True)

        # Set up the logger
        self.logger = logging.getLogger('yasap')
        self.logger.setLevel(logging.DEBUG)

        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        buffer_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # Set up file handler
        log_file = (self.logs_dir /
                    f'yasap_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Set up rolling buffer handler
        self.buffer_handler = RollingBufferHandler()
        self.buffer_handler.setFormatter(buffer_formatter)
        self.logger.addHandler(self.buffer_handler)

        # Set up stderr handler
        stderr_handler = logging.StreamHandler()
        stderr_handler.setLevel(logging.DEBUG)
        stderr_handler.setFormatter(CustomFormatter())
        self.logger.addHandler(stderr_handler)

    def get_log_buffer(self):
        """Get the current log buffer contents."""
        return self.buffer_handler.get_buffer()

    def log(self, level, message):
        """Log a message at the specified level."""
        self.logger.log(level, message)

    def info(self, message):
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message):
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message):
        """Log an error message."""
        self.logger.error(message)

    def debug(self, message):
        """Log a debug message."""
        self.logger.debug(message)
