import logging
import os
from datetime import datetime
from typing import Optional


def build_logger(
    name: str,
    log_dir: str = "logs",
    pipeline_name: str = "general",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Industry-style logger:
    - Console logs for local dev
    - File logs for reproducibility + debugging
    - One log file per run
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if the module is imported multiple times
    if logger.handlers:
        return logger

    os.makedirs(os.path.join(log_dir, pipeline_name), exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, pipeline_name, f"{pipeline_name}_{timestamp}.log")

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    # Helpful line so you know where the file is
    logger.info("Logging to file: %s", log_path)

    return logger
