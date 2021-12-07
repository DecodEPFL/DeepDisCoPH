#!/usr/bin/env python
"""
Plot the norm of the backward sensitivity matrix (BSM) obtained during the training of a distributed H-DNN controller.
Author: Clara Galimberti (clara.galimberti@epfl.ch)
Usage:
python bsm.py                --layer         [LAYER]             \
Flags:
  --layer: Number of the end-layer (k) in (1,N]. Indicates that all gradients are calculated as
           $\frac{\partial \zeta_k}{\partial \zeta_{j}}$ for 0<j<k.
           Set layer=-1 for using the output at layer N.
"""

import argparse
from os.path import isfile
import torch

from plots import plot_grads
from train import train_HDNN_TV


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=-1)
    args = parser.parse_args()
    # Check if bsm file has been already created
    if not(isfile('bsm.pt')):
        print("Creating BSM matrix... (This can take some time)")
        s = torch.eye(12) + torch.diag(torch.ones(12 - 1), 1) + torch.diag(torch.ones(12 - 1), -1)
        s[0, -1] = 1
        s[-1, 0] = 1
        train_HDNN_TV(12, 5, 101, 0.5, s, 1e-2, 0.5, 100, grad_info=True)
        print("BSM matrix created!")
    # Plots gradients
    plot_grads(args.layer, save=True, filename='distributed_HDNN')
