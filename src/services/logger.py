"""
Multi-process safe logging system

Features:
1. Multi-process safe logging
2. Output to both file and console
3. Simple configuration interface
4. Support for custom log levels
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Protocol, Any


# Custom log level definitions
# Format: (level_name, level_value)
# Standard levels: CRITICAL=50, ERROR=40, WARNING=30, INFO=20, DEBUG=10
CUSTOM_LEVELS_TO_REGISTER = [
    ("PredictorDebug", 11),
    ("CacheDebug", 12),
    ("PricingDebug", 13),
    ("MPsolverDebug", 14),
    ("NodeTreeDebug", 15),
    ("mainDebug", 16),
]


def _make_logger_method(level_num):
    """Factory function to create log methods for custom levels."""

    def custom_log_method(self, message, *args, **kws):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kws)

    return custom_log_method


def _register_custom_log_levels():
    """Register all custom log levels."""
    for level_name, level_num in CUSTOM_LEVELS_TO_REGISTER:
        level_name_upper = level_name.upper()
        level_name_lower = level_name.lower()

        if hasattr(logging, level_name_upper):
            continue

        try:
            logging.addLevelName(level_num, level_name_upper)
            log_method = _make_logger_method(level_num)
            setattr(logging.Logger, level_name_lower, log_method)
            setattr(logging, level_name_upper, level_num)

        except Exception as e:
            print(f"Error: Failed to register custom log level {level_name_upper}. Error: {e}")


_register_custom_log_levels()


class CustomLoggerProtocol(Protocol):
    """Type protocol for logger instances with custom methods."""

    def debug(self, msg: object, *args: object, **kwargs: Any) -> None: ...
    def info(self, msg: object, *args: object, **kwargs: Any) -> None: ...
    def warning(self, msg: object, *args: object, **kwargs: Any) -> None: ...
    def error(self, msg: object, *args: object, **kwargs: Any) -> None: ...
    def critical(self, msg: object, *args: object, **kwargs: Any) -> None: ...
    def log(self, level: int, msg: object, *args: object, **kwargs: Any) -> None: ...
    def isEnabledFor(self, level: int) -> bool: ...

    def cachedebug(self, msg: object, *args: object, **kwargs: Any) -> None: ...
    def predictordebug(self, msg: object, *args: object, **kwargs: Any) -> None: ...
    def pricingdebug(self, msg: object, *args: object, **kwargs: Any) -> None: ...
    def mpsolverdebug(self, msg: object, *args: object, **kwargs: Any) -> None: ...
    def nodetreedebug(self, msg: object, *args: object, **kwargs: Any) -> None: ...
    def maindebug(self, msg: object, *args: object, **kwargs: Any) -> None: ...

class LoggingService:
    """Multi-process safe logging service."""

    def __init__(self, log_file="debug.log", level="INFO", log_dir="logs", manager=None):
        """
        Initialize logging service.

        Args:
            log_file: Log file name
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL, or custom)
            log_dir: Log directory
            manager: multiprocessing.Manager instance for shared queue
        """
        if manager is None:
            raise ValueError("LoggingService requires a multiprocessing.Manager instance.")
        
        self.log_file = log_file
        self.level = level.upper()
        self.log_dir = log_dir

        self._log_queue = manager.Queue(-1) 
        self._queue_listener = None
        self._started = False

        self.log_path = Path(log_dir)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.full_log_path = self.log_path / log_file

    def start(self):
        """Start logging system (call in main process)."""
        if self._started:
            print("Warning: Logging service already started.")
            return

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(processName)s-%(process)d] %(levelname)-7s %(name)-20s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(self.full_log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        try:
            console_level = getattr(logging, self.level)
        except AttributeError:
            print(f"Warning: Level '{self.level}' not found. Defaulting to INFO.")
            console_level = logging.INFO

        console_handler.setLevel(console_level)

        self._queue_listener = logging.handlers.QueueListener(self._log_queue, file_handler, console_handler, respect_handler_level=True)
        self._queue_listener.start()

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        self._started = True
        print(f"✓ Logging service started: {self.full_log_path.absolute()}")

    def stop(self):
        """Stop logging system (call at program end)."""
        if not self._started:
            return

        try:
            if self._queue_listener:
                self._queue_listener.stop()
                self._queue_listener = None
                print("✓ Logging service stopped.")
        except Exception as e:
            print(f"Warning: Error while stopping logging service: {e}")
        finally:
            self._started = False

    def get_logger(self, name=None) -> "CustomLoggerProtocol":
        """
        Get logger instance (can be called from any process)

        Args:
            name: logger name, typically pass __name__

        Returns:
            logging.Logger: Configured logger instance (with custom level methods)
        """
        # If no name provided, try to auto-detect calling module name
        if name is None:
            import inspect

            frame = inspect.currentframe().f_back
            name = frame.f_globals.get("__name__", "Unknown")

        logger = logging.getLogger(name)

        # Clear any existing handlers (avoid duplicates)
        if logger.handlers:
            logger.handlers.clear()

        # Add queue handler (all logs sent through queue)
        queue_handler = logging.handlers.QueueHandler(self._log_queue)
        logger.addHandler(queue_handler)

        # --- Critical fix ---
        # Must explicitly set logger level in subprocesses/subthreads.
        # Otherwise, logger in subprocess will default to WARNING level,
        # causing all DEBUG/INFO logs to be discarded before reaching the queue.
        # We set it to DEBUG so it can capture logs of all levels.
        # Actual level filtering is handled by QueueListener's handlers in main process.
        logger.setLevel(logging.DEBUG)

        logger.propagate = False  # Don't propagate to avoid duplicate logging

        return logger

    # Context manager support
    def __enter__(self):
        """Support for with statement"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically stop on exit"""
        self.stop()
        return False

    # Static methods: convenience functions
    @staticmethod
    def log_section(logger, title, level="INFO"):
        """
        Log a section title with separator lines

        Args:
            logger: logger instance
            title: title text
            level: log level (can also use "SUCCESS", "TRACE", etc.)
        """
        separator = "=" * 60
        # Use .lower() to match dynamically added method names
        try:
            log_func = getattr(logger, level.lower())
        except AttributeError:
            print(f"Warning: log function for level '{level}' not found. Defaulting to info.")
            log_func = logger.info

        log_func("")
        log_func(separator)
        log_func(f"  {title}")
        log_func(separator)

    @staticmethod
    def log_dict(logger, data_dict, title="Data", level="DEBUG"):
        """
        Log dictionary content (for debugging)

        Args:
            logger: logger instance
            data_dict: dictionary to log
            title: title
            level: log level
        """
        try:
            log_func = getattr(logger, level.lower())
        except AttributeError:
            print(f"Warning: log function for level '{level}' not found. Defaulting to debug.")
            log_func = logger.debug

        log_func(f"{title}:")
        for key, value in data_dict.items():
            log_func(f"  {key}: {value}")
            
    def __getstate__(self):
        """
        Prepare object for pickling (serialization).
        This removes the non-serializable _queue_listener attribute.
        """
        # Copy instance dictionary
        state = self.__dict__.copy()
        
        # Remove listener as it contains non-serializable thread locks
        # Subprocesses don't need listener, they only need queue
        if '_queue_listener' in state:
            del state['_queue_listener']
            
        return state

    def __setstate__(self, state):
        """
        Restore object after unpickling (deserialization).
        Ensures _queue_listener attribute is initialized to None in subprocess.
        """
        # Restore dictionary
        self.__dict__.update(state)
        
        # Listener only exists in main process
        self._queue_listener = None
