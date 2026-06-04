"""
Logging utilities for experiment tracking.

This module provides utilities for logging experiments,
including file logging and experiment tracking.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


def setup_logger(
    name: str = "vlm_eval",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with console and file output.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level

    Returns:
        Configured logger

    Example:
        >>> logger = setup_logger('experiment', 'logs/exp.log')
        >>> logger.info('Starting experiment')
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "vlm_eval") -> logging.Logger:
    """Get existing logger or create new one."""
    return logging.getLogger(name)


class ExperimentLogger:
    """
    Logger for tracking experiments and results.

    Example:
        >>> exp_logger = ExperimentLogger('logs/experiments')
        >>> exp_logger.log_experiment('exp1', {'accuracy': 0.95})
    """

    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'experiments_{timestamp}.jsonl'

    def log_experiment(self, experiment_name: str, results: dict):
        """Log experiment results to JSONL file."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'experiment': experiment_name,
            'results': results
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
