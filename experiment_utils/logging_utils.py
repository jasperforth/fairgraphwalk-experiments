import logging
import sys
import threading
from datetime import datetime
from pathlib import Path

_log_lock = threading.Lock()
_shared_file_handler = None

def setup_shared_file_handler(log_dir: Path):
    global _shared_file_handler
    if _shared_file_handler is None:
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"logfiles/logs_{current_date}/experiment_log.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _shared_file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        _shared_file_handler.setFormatter(formatter)
    return _shared_file_handler

def setup_main_logging(log_dir: Path):
    shared_file_handler = setup_shared_file_handler(log_dir)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[shared_file_handler, logging.StreamHandler()])
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    logger.info("Logging is set up.")
    return logger

def setup_worker_logging(name="worker", log_dir: Path = None):
    with _log_lock:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Create a StreamHandler if not already present
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(stream_formatter)
            logger.addHandler(stream_handler)
        
        # Add the shared FileHandler
        if log_dir:
            shared_file_handler = setup_shared_file_handler(log_dir)
            if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
                logger.addHandler(shared_file_handler)
        
        logger.propagate = False
    return logger

