"""Helper class to manage threads"""

from Queue import Queue
import threading

class ThreadManager(object):
    """Thread manager"""
    # pylint: disable=too-few-public-methods
    def __init__(self, num_threads):
        self.network_count = 0
        self.train_err = 0
        self.updating = 0
        self.updates_cv = threading.Condition()
        self.err_lock = threading.Lock()
        self.pool = Queue(num_threads)
