import logging
import logging.handlers
import multiprocessing
import sys
from pathlib import Path

# Global singletons
_log_queue = None
_listener = None

def get_log_queue():
    """
    Returns the same queue object in both main and child processes,
    because this function is imported and called at runtime in each.
    """
    global _log_queue
    if _log_queue is None:
        # Create one single queue. On Windows or with 'spawn', each process
        # will run this function independently, but we want exactly one
        # queue object in the main process. So typically we do this in
        # an if __name__ == '__main__': block or ensure joblib "fork" mode
        _log_queue = multiprocessing.Queue(-1)
    return _log_queue

def setup_queue_listener(log_dir: Path, tag: str = "main"):
    """
    Sets up the QueueListener in the main process only,
    adding both file and stream handlers.
    """
    global _listener
    if _listener is not None:
        # Already set up, just return it
        return _listener

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "experiment_log.log"

    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # The QueueListener will pull from the queue and broadcast to these handlers
    _listener = logging.handlers.QueueListener(
        get_log_queue(), file_handler, stream_handler, respect_handler_level=True
    )
    _listener.start()
    return _listener

def stop_listener():
    """
    Called by the main process after all parallel jobs finish,
    so that all queued log records are flushed and listener is shut down.
    """
    global _listener
    if _listener:
        _listener.stop()
        _listener = None

def setup_logger_for_process(logger_name="default"):
    """
    In both main and child processes, attach a QueueHandler so that
    .info(...) calls get sent into our global log queue.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # Clear old handlers to avoid duplicates
    logger.handlers.clear()

    q_handler = logging.handlers.QueueHandler(get_log_queue())
    logger.addHandler(q_handler)
    logger.propagate = False
    return logger
