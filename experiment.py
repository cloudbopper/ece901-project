"""Top-level script to run experiments"""

import argparse
import pickle
import subprocess
import matplotlib
import matplotlib.pyplot as plt

def main():
    """Run experiments"""
    # pylint: disable=too-many-locals,unused-variable,line-too-long,too-many-branches,too-many-statements
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", required=True,
                        choices=["learningcurves", "speedup_time", "batch_downsizing", "worker_iterations"])

    parser.add_argument("-batch_size", default=500, type=int)
    parser.add_argument("-dropout_rate", type=float, default=0.5)
    parser.add_argument("-worker_iterations", type=int, default=1)
    parser.add_argument("-num_epochs", default=10, type=int)
    parser.add_argument("-term_val_acc", default=90., type=float)

    args = parser.parse_args()
    font = {'weight' : 'bold',
            'size'   : 16}
    matplotlib.rc('font', **font)
    dropout_types = ["regular", "disjoint", "overlapping", "overlapping -synchronize_workers"]
    output_header = "data/1/%s" % args.type
    if args.type == "learningcurves":
        for dropout_type in dropout_types:
            plt.figure(figsize=(16, 12))
            for num_threads in range(1, 5):
                cmd = ("python pipeline.py -threads %d -batch_size %d -dropout_type %s "
                       "-dropout_rate %f -num_epochs %d" %
                       (num_threads, args.batch_size, dropout_type, args.dropout_rate,
                        args.num_epochs))
                print "Cmd: %s" % cmd
                obj = subprocess.Popen(cmd, shell=True)
                obj.wait()
                with open("data/objs.pickle") as objs_file:
                    epoch_times, train_losses, val_pc_accs, test_pc_acc = pickle.load(objs_file)
                plt.plot(range(1, args.num_epochs + 1), val_pc_accs,
                         label=("Number of threads: %d" % (num_threads)))
            plt.xlabel("Number of epochs")
            plt.ylabel("Validation set accuracy")
            plt.xlim([0, args.num_epochs + 1])
            plt.ylim([0., 100.])
            plt.title("Learning curves for dropout type: %s" % dropout_type)
            plt.legend(loc='lower right')
            plt.savefig("%s_%s" % (output_header, dropout_type))
    elif args.type == "speedup_time":
        max_time = 0.
        plt.figure(figsize=(16, 12))
        for dropout_type in dropout_types:
            times = []
            for num_threads in range(1, 5):
                cmd = ("python pipeline.py -threads %d -batch_size %d -dropout_type %s "
                       "-dropout_rate %f -term_val_acc %f" %
                       (num_threads, args.batch_size, dropout_type, args.dropout_rate,
                        args.term_val_acc))
                print "Cmd: %s" % cmd
                obj = subprocess.Popen(cmd, shell=True)
                obj.wait()
                with open("data/objs.pickle") as objs_file:
                    epoch_times, train_losses, val_pc_accs, test_pc_acc = pickle.load(objs_file)
                times.append(sum(epoch_times))
                max_time = max(max_time, times[-1])
            plt.plot(range(1, 5), times, label=("Dropout type: %s" % dropout_type))
        plt.xlabel("Number of threads")
        plt.ylabel("Time (seconds)")
        plt.xlim([0, 6])
        plt.ylim([0, max_time + 10.])
        plt.title("Time to reach validation set accuracy of %f" % args.term_val_acc)
        plt.legend(loc='lower right')
        plt.savefig(output_header)
    elif args.type == "batch_downsizing":
        for dropout_type in dropout_types:
            plt.figure(figsize=(16, 12))
            for num_threads in range(1, 5):
                batch_size = args.batch_size / num_threads
                cmd = ("python pipeline.py -threads %d -batch_size %d -dropout_type %s "
                       "-dropout_rate %f -num_epochs %d" %
                       (num_threads, batch_size, dropout_type, args.dropout_rate,
                        args.num_epochs))
                print "Cmd: %s" % cmd
                obj = subprocess.Popen(cmd, shell=True)
                obj.wait()
                with open("data/objs.pickle") as objs_file:
                    epoch_times, train_losses, val_pc_accs, test_pc_acc = pickle.load(objs_file)
                plt.plot(range(1, args.num_epochs + 1), val_pc_accs,
                         label=("Number of threads: %d" % (num_threads)))
            plt.xlabel("Number of epochs")
            plt.ylabel("Validation set accuracy")
            plt.xlim([0, args.num_epochs + 1])
            plt.ylim([0., 100.])
            plt.title("Learning curves for dropout type: %s with thread-count adjusted batch sizes" % dropout_type)
            plt.legend(loc='lower right')
            plt.savefig("%s_%s" % (output_header, dropout_type))
    elif args.type == "worker_iterations":
        worker_iterations = [1, 5, 10]
        for dropout_type in dropout_types:
            for num_threads in range(1, 5):
                plt.figure(figsize=(16, 12))
                for worker_iteration in worker_iterations:
                    cmd = ("python pipeline.py -threads %d -batch_size %d -dropout_type %s "
                           "-dropout_rate %f -num_epochs %d -worker_iterations %d" %
                           (num_threads, args.batch_size, dropout_type, args.dropout_rate,
                            args.num_epochs, worker_iteration))
                    print "Cmd: %s" % cmd
                    obj = subprocess.Popen(cmd, shell=True)
                    obj.wait()
                    with open("data/objs.pickle") as objs_file:
                        epoch_times, train_losses, val_pc_accs, test_pc_acc = pickle.load(objs_file)
                    plt.plot(range(1, args.num_epochs + 1), val_pc_accs,
                             label=("Worker iteration count: %d" % worker_iteration))
                plt.xlabel("Number of epochs")
                plt.ylabel("Validation set accuracy")
                plt.xlim([0, args.num_epochs + 1])
                plt.ylim([0., 100.])
                plt.title("Learning curves for dropout type: %s for %d threads with various values of worker iterations" % (dropout_type, num_threads))
                plt.legend(loc='lower right')
                plt.savefig("%s_%s_%d" % (output_header, dropout_type, num_threads))

if __name__ == "__main__":
    main()
