"""Read-only bundle assets and writable per-user diagnostics."""
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import tempfile


def resource_path(relative):
    root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
    return root / relative


def configure_logging(name):
    logger = logging.getLogger()
    if sys.platform == 'darwin':
        primary = Path.home() / 'Library' / 'Logs' / 'Alpha2048'
    else:
        primary = Path(os.environ.get('LOCALAPPDATA') or Path.home()) / 'Alpha2048' / 'logs'
    for base in (primary, Path(tempfile.gettempdir()) / 'Alpha2048' / 'logs'):
        try:
            base.mkdir(parents=True, exist_ok=True)
            path = base / f'{name}.log'
            handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=2, encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.ERROR)
            return path
        except OSError:
            continue
    # Diagnostics must not prevent an otherwise usable app from starting.
    return None


def install_error_handler(root, name):
    path = configure_logging(name)
    def callback_error(kind, value, traceback):
        logging.error('%s UI failed', name, exc_info=(kind,value,traceback))
        from tkinter import messagebox
        message = str(value)
        if path is not None:
            message += f'\n\nLog: {path}'
        messagebox.showerror('2048', message, parent=root)
    root.report_callback_exception = callback_error
    return path
