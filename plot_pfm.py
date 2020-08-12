import numpy as np
import matplotlib.pyplot as plt
import argparse

from utils.io_utils import load_pfm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    args = parser.parse_args()

    img = load_pfm(args.file)
    plt.imshow(img)
    plt.show()
