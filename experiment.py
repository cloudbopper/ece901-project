"""Top-level script to run experiments"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pipeline import pipeline

def main():
    """Run experiments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", required=True,
                        choices=["learningcurves"])
    parser.add_argument("-output_filename", required=True)
    parser.add_argument("-title", required=True)
    parser.add_argument("-num_trials", default=10, type=int)

    parser.add_argument("-batch_size", default=500, type=int)
    parser.add_argument("-dropout_type", help="type of dropout", required=True,
                        choices=["disjoint", "overlapping"])
    parser.add_argument("-input_dropout_rate", type=float, default=0.2)
    parser.add_argument("-dropout_rate", type=float, default=0.5)
    parser.add_argument("-synchronize_workers", action="store_true")
    parser.add_argument("-worker_iterations", type=int, default=1)
    parser.add_argument("-num_epochs", default=20, type=int)
    parser.add_argument("-term_val_acc", default=100, type=int)
    parser.add_argument("-debug", action="store_true")

    args = parser.parse_args()

    if args.type == "learningcurves":
        for num_threads in range(1, 5):
            args.threads = num_threads
            epoch_times_avg = np.zeros(args.num_epochs)
            train_losses_avg = np.zeros(args.num_epochs)
            val_pc_accs_avg = np.zeros(args.num_epochs)
            test_pc_acc_avg = 0.
            plt.figure(num=1, figsize=(16, 12))
            for _ in range(args.num_trials):
                epoch_times, train_losses, val_pc_accs, test_pc_acc = pipeline(args)
                epoch_times_avg += epoch_times
                train_losses_avg += train_losses
                val_pc_accs_avg += val_pc_accs
                test_pc_acc_avg += test_pc_acc
            epoch_times_avg /= args.num_trials
            train_losses_avg /= args.num_trials
            val_pc_accs_avg /= args.num_trials
            test_pc_acc_avg /= args.num_trials
            plt.plot(train_losses_avg, range(1, args.num_epochs + 1),
                     label=("Number of threads: %d" % (num_threads)))
            plt.title(args.title)
            plt.legend(loc='upper right')
            plt.savefig(args.output_filename)


if __name__ == "__main__":
    main()
