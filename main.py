"""Multithreaded dropout experiment"""

import argparse
from collections import deque
import os
import time
import threading
import urllib
#from urllib import urlretrieve
from urllib.request import urlretrieve

import lasagne
import numpy as np
import theano
import theano.tensor as T

import dropout
from sgd import sgd
from thread_manager import TM

# pylint: disable=superfluous-parens

class WorkerThread(threading.Thread):
    """Worker thread implementation"""
    def __init__(self, read_fn, write_fn, train_fn, tid, inputs, targets):
        # pylint: disable=too-many-arguments
        threading.Thread.__init__(self)
        self.tid = tid
        self.train_fn = train_fn
        self.read_fn = read_fn
        self.write_fn = write_fn
        self.inputs = inputs
        self.targets = targets

    def run(self):
        # Read per-thread params from global params
        TM.updates_cv.acquire()
        while TM.updating:
            TM.updates_cv.wait()
        TM.updating = 1
        TM.updates_cv.release()

        self.read_fn(self.tid)

        TM.updates_cv.acquire()
        assert TM.updating # TODO: remove
        TM.updating = 0
        TM.updates_cv.notify()
        TM.updates_cv.release()

        # Train and update overall training error
        out = self.train_fn(self.inputs, self.targets)
        TM.err_lock.acquire()
        TM.train_err += out
        TM.err_lock.release()

        # Write per-thread params into global params
        TM.updates_cv.acquire()
        while TM.updating:
            TM.updates_cv.wait()
        TM.updating = 1
        TM.updates_cv.release()

        self.write_fn(self.tid)

        TM.updates_cv.acquire()
        assert TM.updating # TODO: remove
        TM.updating = 0
        TM.updates_cv.notify()
        TM.updates_cv.release()

        # Remove worker from active pool
        TM.worker_cv.acquire()
        TM.num_workers -= 1
        TM.worker_cv.notify()
        TM.worker_cv.release()


def main():
    """Command-line pipeline"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-threads", help="number of threads to launch",
                        required=True, type=int)
    parser.add_argument("-worker_iterations", help="number of iterations to run"
                        " each worker for", required=True, type=int)
    parser.add_argument("-batch_size", help="size of batch operated upon by "
                        "each worker thread", required=True, type=int)
    parser.add_argument("-dropout_type", help="type of dropout", required=True,
                        choices=["disjoint", "overlapping"])
    parser.add_argument("-dropout_rate", required=True)

    parser.add_argument("-num_epochs", default=500, type=int)

    args = parser.parse_args()
    pipeline(args)



def pipeline(args):
    """Dropout experiment pipeline
    Note: currently only implements overlapping dropout
    """
    # TODO: fragment function more
    # pylint: disable=too-many-locals,invalid-name
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Create neural network model
    print("Building model and compiling functions...")
    read_fn, write_fn, train_fns, val_fn = gen_computational_graphs(args)

    # Finally, launch the training loop.
    print("Starting training...")
    for epoch in range(args.num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            train_batches += 1
            inputs, targets = batch
            TM.worker_cv.acquire()
            while TM.num_workers >= args.threads:
                TM.worker_cv.wait()
            TM.num_workers += 1
            TM.worker_cv.release()
            tid, train_fn = train_fns.popleft()
            train_fns.append((tid, train_fn))
            worker = WorkerThread(read_fn, write_fn, train_fn, tid, inputs, targets)
            worker.start()
        TM.worker_cv.acquire()
        while TM.num_workers > 0:
            TM.worker_cv.wait()
        TM.worker_cv.release()
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, args.batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, args.num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(TM.train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))


def gen_computational_graphs(args):
    """Generates Theano functions for training/testing the network"""
    # pylint: disable=too-many-locals
    # Create separate training networks for each thread
    train_fns = deque()
    params_per_thread = []
    for tid in range(args.threads):
        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs_%d' % tid)
        target_var = T.ivector('targets_%d' % tid)
        network = build_mlp(input_var)

        # Create a loss expression for training
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()

        # Create update expressions for training
        # TODO: Backprop should run without locking, only writing the updates should need locking
        params = lasagne.layers.get_all_params(network, trainable=True)
        params_per_thread.append(params)
        updates = sgd(loss, params, learning_rate=0.01)

        # Compile a function performing a training step on a mini-batch
        # and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)
        train_fns.append((tid, train_fn))

    # Compile a second function computing the validation loss and accuracy:
    input_var = T.tensor4('input')
    target_var = T.ivector('targets')
    network = build_mlp(input_var)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # update params using per-thread params
    gparams = lasagne.layers.get_all_params(network, trainable=True)
    def read_fn(tid):
        """Update per-thread params from global params"""
        tparams = params_per_thread[tid]
        for idx, _ in enumerate(gparams):
            tparams[idx].set_value(gparams[idx].get_value())
    def write_fn(tid):
        """Update global params from per-thread params"""
        tparams = params_per_thread[tid]
        for idx, _ in enumerate(gparams):
            gparams[idx].set_value(tparams[idx].get_value())
    return read_fn, write_fn, train_fns, val_fn


def load_dataset():
    """Loads dataset"""
    # pylint: disable=invalid-name
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        """Downloader"""
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        """Loads MNIST image data"""
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        """Loads MNIST labels"""
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """Helper function to iterate over training data in mini-batches"""
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_mlp(input_var=None):
    """Build MLP model"""
    # pylint: disable=bad-continuation
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = dropout.DropoutLayerOverlapping(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = dropout.DropoutLayerOverlapping(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = dropout.DropoutLayerOverlapping(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

if __name__ == "__main__":
    main()
