"""SGD"""

from collections import OrderedDict

import lasagne
from thread_manager import TM

def sgd(loss_or_grads, params, learning_rate):
    """Stochastic Gradient Descent (SGD) updates

    Generates update expressions of the form:

    * ``param := param - learning_rate * gradient``

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()


    # Update params in thread-safe manner
    # TM.updates_cv.acquire()
    # while TM.updating:
    #     TM.updates_cv.wait()
    # TM.updating = 1
    # TM.updates_cv.release()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    # TM.updates_cv.acquire()
    # assert TM.updating # TODO: remove
    # TM.updating = 0
    # TM.updates_cv.notify()
    # TM.updates_cv.release()


    return updates
