"""
Download N-MNIST event camera dataset using tonic.
Usage: python download.py [--test]
"""

import argparse
import os
import tonic

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description='Download N-MNIST dataset')
    parser.add_argument('--test', action='store_true', help='Also download test set')
    args = parser.parse_args()

    print(f"Downloading N-MNIST train set to {SAVE_DIR}/NMNIST/ ...")
    train = tonic.datasets.NMNIST(save_to=SAVE_DIR, train=True)
    print(f"Train set: {len(train)} samples")

    if args.test:
        print(f"Downloading N-MNIST test set to {SAVE_DIR}/NMNIST/ ...")
        test = tonic.datasets.NMNIST(save_to=SAVE_DIR, train=False)
        print(f"Test set: {len(test)} samples")

    print("Done.")


if __name__ == "__main__":
    main()
