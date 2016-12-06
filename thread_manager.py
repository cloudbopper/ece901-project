"""Helper class to manage threads"""

import threading

class TM(object):
    """Thread manager"""
    # pylint: disable=too-few-public-methods
    num_workers = 0
    train_err = 0
    updating = 0
    worker_cv = threading.Condition()
    updates_cv = threading.Condition()
    err_lock = threading.Lock()
