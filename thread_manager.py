"""Helper class to manage threads"""

from Queue import Queue
import threading

class TM(object):
    """Thread manager"""
    # pylint: disable=too-few-public-methods
    network_count = 0
    train_err = 0
    updating = 0
    updates_cv = threading.Condition()
    err_lock = threading.Lock()

    @staticmethod
    def init_pool(num_threads):
        """Initialize synchronized worker task queue"""
        TM.pool = Queue(num_threads)
