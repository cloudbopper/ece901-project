"""Top-level script to run experiments"""

import argparse
import pickle
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def main():
    """Run experiments"""
    # pylint: disable=too-many-locals
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", required=True,
                        choices=["learningcurves", "speedup_time"])
    parser.add_argument("-output_filename", required=True)
    parser.add_argument("-title", required=True)
    parser.add_argument("-num_trials", default=1, type=int)

    parser.add_argument("-batch_size", default=500, type=int)
    parser.add_argument("-dropout_type", help="type of dropout", required=True,
                        choices=["disjoint", "overlapping"])
    parser.add_argument("-dropout_rate", type=float, default=0.5)
    parser.add_argument("-synchronize_workers", action="store_true")
    parser.add_argument("-worker_iterations", type=int, default=1)
    parser.add_argument("-num_epochs", default=10, type=int)
    parser.add_argument("-term_val_acc", default=100., type=float)

    args = parser.parse_args()

    syncworkers = ""
    if args.synchronize_workers:
        syncworkers = " -synchronize_workers"
    if args.type == "learningcurves":
        args.term_val_acc = 100.

    font = {'weight' : 'bold',
            'size'   : 30}
    matplotlib.rc('font', **font)
    plt.figure(num=1, figsize=(16, 12))
    for num_threads in range(1, 2):
        args.threads = num_threads
        epoch_times_avg = np.zeros(args.num_epochs)
        train_losses_avg = np.zeros(args.num_epochs)
        val_pc_accs_avg = np.zeros(args.num_epochs)
        test_pc_acc_avg = 0.
        for _ in range(args.num_trials):
            cmd = ("python pipeline.py -threads %d -batch_size %d -dropout_type %s "
                   "-dropout_rate %f -num_epochs %d -term_val_acc %f %s" %
                   (num_threads, args.batch_size, args.dropout_type, args.dropout_rate,
                    args.num_epochs, args.term_val_acc, syncworkers))
            print "Cmd: %s" % cmd
            obj = subprocess.Popen(cmd, shell=True)
            obj.wait()
            with open("data/objs.pickle") as objs_file:
                epoch_times, train_losses, val_pc_accs, test_pc_acc = pickle.load(objs_file)
            epoch_times_avg += epoch_times
            train_losses_avg += train_losses
            val_pc_accs_avg += val_pc_accs
            test_pc_acc_avg += test_pc_acc
        epoch_times_avg /= args.num_trials
        train_losses_avg /= args.num_trials
        val_pc_accs_avg /= args.num_trials
        test_pc_acc_avg /= args.num_trials
        if args.type == "learningcurves":
            plt.plot(range(1, args.num_epochs + 1), val_pc_accs_avg,
                     label=("Number of threads: %d" % (num_threads)))
    if args.type == "learningcurves":
        plt.xlabel("Number of epochs")
        plt.ylabel("Validation set accuracy")
        plt.xlim([0, args.num_epochs + 1])
        plt.ylim([0., 100.])
    plt.title(args.title)
    plt.legend(loc='lower right')
    plt.savefig(args.output_filename)


if __name__ == "__main__":
    main()
