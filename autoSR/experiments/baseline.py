
import os
import time
from argparse import ArgumentParser

from baselines.train import train_test

DATADIR = os.environ["DATADIR"]
DATASETSDIR = os.environ["DATASETSDIR"]
RESULTSDIR = os.environ["RESULTSDIR"]

if __name__ == "__main__":
    start = time.time()
    parser = ArgumentParser()
    parser.add_argument("--baseline", type=str, choices=['wdsr', 'rcan','rcan_undeep'])
    parser.add_argument("--dataset", type=str,
                        choices=['cerrado', 'sr_ucmerced', 'oli2msi', 'sr_so2sat','sent_nicfi'])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--id", type=str)
    args = parser.parse_args()

    train_test(args.baseline, args.dataset, args.epochs, args.id)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
