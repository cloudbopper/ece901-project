"""Multithreaded dropout experiment"""

import argparse
import os
import pickle
import time
import threading
from urllib import urlretrieve

import lasagne
import numpy as np
import theano
import theano.tensor as T

import dropout
from threadmanager import ThreadManager

# pylint: disable=superfluous-parens


class WorkerThread(threading.Thread):
    """Worker thread implementation"""
    # pylint: disable=no-self-use,too-many-instance-attributes,too-many-arguments
    def __init__(self, tid, read_fn, write_fn, train_fn, thread_manager):
        threading.Thread.__init__(self)
        self.tid = tid
        self.read_fn = read_fn
        self.write_fn = write_fn
        self.train_fn = train_fn
        self.thread_manager = thread_manager
        self.daemon = True
        self.initial_weights = None
        self.tparams = None
        self.increments = None
        self.age = 0

    def pre_update(self):
        """Pre-update CV management"""
        self.thread_manager.updates_cv.acquire()
        while self.thread_manager.updating:
            self.thread_manager.updates_cv.wait()
        self.thread_manager.updating = 1
        self.thread_manager.updates_cv.release()

    def post_update(self):
        """Post-update CV management"""
        self.thread_manager.updates_cv.acquire()
        self.thread_manager.updating = 0
        self.thread_manager.updates_cv.notify()
        self.thread_manager.updates_cv.release()

    def read_params(self):
        """Read per-thread params from global params"""
        self.pre_update()
        self.tparams = self.read_fn(self.tid)
        self.post_update()
        self.initial_weights = [np.copy(tparam.get_value()) for tparam in self.tparams]

    def write_params(self):
        """Write per-thread params into global params"""
        self.increments = ([self.tparams[idx].get_value() - self.initial_weights[idx]
                            for idx, _ in enumerate(self.tparams)])
        self.pre_update()
        self.write_fn(self.increments)
        self.post_update()

    def train(self, inputs, targets, dropout_mask):
        """Train and update overall training error"""
        out = self.train_fn(self.tid)(inputs, targets, *dropout_mask)
        self.thread_manager.err_lock.acquire()
        self.thread_manager.train_err += out
        self.thread_manager.err_lock.release()

    def run(self):
        """Run worker thread"""
        while True:
            self.age += 1
            batches, dropout_mask = self.thread_manager.pool.get()
            self.read_params()
            for inputs, targets in batches:
                self.train(inputs, targets, dropout_mask)
            self.write_params()
            self.thread_manager.pool.task_done()


def main():
    """Command-line pipeline"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-threads", help="number of threads to launch",
                        required=True, type=int)
    parser.add_argument("-batch_size", help="size of batch operated upon by "
                        "each worker thread", required=True, type=int)
    parser.add_argument("-dropout_type", help="type of dropout", required=True,
                        choices=["disjoint", "overlapping"])
    parser.add_argument("-input_dropout_rate", type=float, default=0.2)
    parser.add_argument("-dropout_rate", type=float, default=0.5)
    parser.add_argument("-synchronize_workers", help="master waits for all workers to "
                        "finish the last round of minibatch SGD before starting another "
                        "round; always enabled for disjoint dropout", action="store_true")
    parser.add_argument("-debug", action="store_true")
    parser.add_argument("-worker_iterations", help="number of iterations to run on a single worker"
                        " with a fixed network", type=int, default=1)
    parser.add_argument("-num_epochs", default=500, type=int)
    parser.add_argument("-term_val_acc", help="level of validation set accuracy in % after "
                        "reaching which algorithm will terminate", default=100., type=float)

    args = parser.parse_args()
    epoch_times, train_losses, val_pc_accs, test_pc_acc = pipeline(args)
    with open("data/objs.pickle", "w") as objs_file:
        pickle.dump([epoch_times, train_losses, val_pc_accs, test_pc_acc], objs_file)

def pipeline(args):
    """Dropout experiment pipeline
    Note: currently only implements overlapping dropout
    """
    # TODO: fragment function more
    # pylint: disable=too-many-locals,invalid-name,too-many-statements,too-many-branches
    theano.config.exception_verbosity = "high"
    if args.dropout_type == "disjoint":
        args.synchronize_workers = True
        args.dropout_rate = 1 - 1./args.threads

    thread_manager = ThreadManager(args.threads)
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Create neural network model
    print("Building model and compiling functions...")
    read_fn, write_fn, train_fn, val_fn, network = gen_computational_graphs(args)
    mask_fn = gen_mask_functions(args, network)

    # Create worker threads
    workers = [None] * args.threads
    for tid in range(args.threads):
        worker = WorkerThread(tid, read_fn, write_fn, train_fn, thread_manager)
        worker.start()
        workers[tid] = worker

    # Finally, launch the training loop.
    print("Starting training...")
    epoch_times = []
    train_losses = []
    val_pc_accs = []

    for epoch in range(args.num_epochs):
        # In each epoch, we do a full pass over the training data:
        start_time = time.time()
        thread_manager.train_err = 0
        train_batches = 0
        batch_generator = iterate_minibatches(X_train, y_train, args.batch_size, shuffle=True)
        epoch_complete = False
        while True:
            dropout_masks = mask_fn()
            for tid in range(args.threads):
                tbatches = []
                for _ in range(args.worker_iterations):
                    try:
                        batch = batch_generator.next()
                        tbatches.append(batch)
                        train_batches += 1
                    except StopIteration:
                        epoch_complete = True
                        break
                if tbatches:
                    thread_manager.pool.put((tbatches, dropout_masks[tid]))
                if epoch_complete:
                    break
            if epoch_complete:
                break
            if args.synchronize_workers:
                thread_manager.pool.join()
                # Validate disjoint dropout
                if args.debug:
                    assert validate_disjoint_dropout(args, workers)
        thread_manager.pool.join()
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
        epoch_times.append(time.time() - start_time)
        train_losses.append(thread_manager.train_err / train_batches)
        val_pc_accs.append(val_acc / val_batches * 100)

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, args.num_epochs, epoch_times[-1]))
        print("  training loss:\t\t{:.6f}".format(train_losses[-1]))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_pc_accs[-1]))

        # Early stopping
        if val_pc_accs[-1] > args.term_val_acc:
            break

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_pc_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1

    test_pc_acc = (test_acc / test_batches * 100)
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_pc_acc))
    return epoch_times, train_losses, val_pc_accs, test_pc_acc


def validate_disjoint_dropout(args, workers):
    """Validate disjoint dropout producing disjoint updates"""
    if args.dropout_type != "disjoint":
        return True
    diffs = []
    for worker in workers:
        diffs.append(worker.increments)
    for idx1, _ in enumerate(workers):
        for idx2, _ in enumerate(workers):
            if idx1 == idx2 or workers[idx1].age != workers[idx2].age:
                # Introduced age due to condition where 1 worker does the work of
                # multiple workers in one master iteration
                continue
            for lidx in range(len(diffs[0]) - 1):
                # Last set of parameters corresponds to bias weights on output layer:
                # these will not be disjoint, so ignore
                if np.count_nonzero(np.multiply(diffs[idx1][lidx], diffs[idx2][lidx])) > 0:
                    return False
    return True

def gen_computational_graphs(args):
    """Generates Theano functions for training/testing the network"""
    # pylint: disable=too-many-locals
    # Create separate training networks for each thread
    train_fns = []
    params_per_thread = []
    netid = 0
    for tid in range(args.threads):
        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs_%d' % tid)
        target_var = T.ivector('targets_%d' % tid)
        network, masks = build_mlp(args, netid, input_var, mask_inputs=True)
        netid += 1

        # Create a loss expression for training
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()

        # Create update expressions for training
        params = lasagne.layers.get_all_params(network, trainable=True)
        params_per_thread.append(params)
        updates = lasagne.updates.sgd(loss, params, learning_rate=0.01)
        # updates = update.nesterov_momentum(loss, params,
        #                                    learning_rate=0.01,
        #                                    momentum=0.9)

        # Compile a function performing a training step on a mini-batch
        # and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var] + masks,
                                   loss, updates=updates, on_unused_input='ignore')
        train_fns.append(train_fn)
    train_fn = lambda tid: train_fns[tid]

    # Compile a second function computing the validation loss and accuracy:
    input_var = T.tensor4('input')
    target_var = T.ivector('targets')
    network, _ = build_mlp(args, netid, input_var)
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
        return tparams
    def write_fn(increments):
        """Update global params from worker increments"""
        for idx, _ in enumerate(gparams):
            gparams[idx].set_value(gparams[idx].get_value() + increments[idx])
    return read_fn, write_fn, train_fn, val_fn, network


def gen_mask_functions(args, network):
    """Returns dropout mask-generating functions"""
    mask_fns = []
    layers = lasagne.layers.get_all_layers(network)
    if args.dropout_type == "overlapping" or args.threads == 1:
        for layer in layers:
            if isinstance(layer, dropout.DropoutLayer):
                # workaround to set mask shape correctly regardless of input layer
                if isinstance(layer.input_layer, lasagne.layers.InputLayer):
                    mask_shape = layer.input_layer.output_shape[1:]
                else:
                    mask_shape = (layer.input_layer.num_units,)
                # pylint: disable=cell-var-from-loop
                func = lambda p=layer.p, shape=mask_shape: np.random.binomial(1, (1-p), shape)
                mask_fns.append(func)
        return lambda: [[func() for func in mask_fns] for _ in range(args.threads)]
    else:
        # Disjoint dropout
        retain_prob = 1./args.threads
        retain_prob = min(retain_prob, 1 - args.dropout_rate)
        mask_fns = []
        for layer in layers:
            if isinstance(layer, dropout.DropoutLayer):
                if isinstance(layer.input_layer, lasagne.layers.InputLayer):
                    # overlaps in input layer OK - parameters should still be disjoint
                    mask_shape = layer.input_layer.output_shape[1:]
                    tfs = [None] * args.threads
                    for tid in range(args.threads):
                        tfs[tid] = lambda p=layer.p, s=mask_shape: np.random.binomial(1, (1-p), s)
                    # pylint: disable=cell-var-from-loop
                    mask_fns.append(lambda: [tfs[tid]() for tid in range(args.threads)])
                else:
                    mask_shape = (layer.input_layer.num_units,)
                    def func(shape=mask_shape):
                        """Computes mask for this layer for all threads"""
                        mask_all = np.random.multinomial(1, [retain_prob] * args.threads, shape)
                        masks = [None] * args.threads
                        for tid in range(args.threads):
                            masks[tid] = mask_all[:, tid]
                        return masks
                    mask_fns.append(func)
        return lambda: np.transpose([func() for func in mask_fns])


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


def build_mlp(args, netid, input_var=None, mask_inputs=False):
    """Build MLP model"""
    # pylint: disable=bad-continuation
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:

    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var,
                                     name="%d_%s" % (netid, "l_in"))

    mask_in = None
    if mask_inputs:
        mask_in = T.ltensor3()
    # Apply 20% dropout to the input data:
    l_in_drop = dropout.DropoutLayer(l_in, mask=mask_in, p=args.input_dropout_rate,
                                     name="%d_%s" % (netid, "l_in_drop"))

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=200,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),
            name="%d_%s" % (netid, "l_hid1"))

    # We'll now add dropout of 50%:
    mask_hid1 = None
    if mask_inputs:
        mask_hid1 = T.lvector()
    l_hid1_drop = dropout.DropoutLayer(l_hid1, mask=mask_hid1, p=args.dropout_rate,
                                       name="%d_%s" % (netid, "l_hid1_drop"))

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=200,
            nonlinearity=lasagne.nonlinearities.rectify,
            name="%d_%s" % (netid, "l_hid2"))

    # 50% dropout again:
    mask_hid2 = None
    if mask_inputs:
        mask_hid2 = T.lvector()
    l_hid2_drop = dropout.DropoutLayer(l_hid2, mask=mask_hid2, p=args.dropout_rate,
                                       name="%d_%s" % (netid, "l_hid2_drop"))

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax,
            name="%d_%s" % (netid, "l_out"))

    masks = [mask_in, mask_hid1, mask_hid2]
    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out, masks

if __name__ == "__main__":
    main()
