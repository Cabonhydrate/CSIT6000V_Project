"""Compact diagnostics for the single-user demo: console + rotating local file."""
import logging
from logging.handlers import RotatingFileHandler
from .config import settings


def get_logger(name: str) -> logging.Logger:
    root = logging.getLogger('paper_tutor')
    if not root.handlers:
        log_dir = settings.data_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(name)s | %(message)s', '%Y-%m-%d %H:%M:%S')
        for handler in (
            logging.StreamHandler(),
            RotatingFileHandler(log_dir / 'app.log', maxBytes=5_000_000, backupCount=2, encoding='utf-8'),
        ):
            handler.setFormatter(formatter)
            root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return root.getChild(name)
