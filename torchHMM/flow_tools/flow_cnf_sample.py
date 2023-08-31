import argparse
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_moons


from cnf_utils import build_model_tabular, standard_normal_logprob
import pandas as pd


def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")

    parser.add_argument(
        "--n",
        default="1000",
        required=False,
        help="nr of   data to sample from flow (default: %(default)s)",
    )

    parser.add_argument(
        "--input-model",
        default="flow_model.pt",
        required=False,
        help="input model (default: %(default)s)",
    )
    parser.add_argument(
        "--input-train-data",
        default="train_data.pkl",
        required=False,
        help="input train data  (just to draw it, default: %(default)s)",
    )

    args = parser.parse_args()

    return args.n, args.input_model, args.input_train_data


def sample_flow(cnf, size, device):
    y = torch.randn(*size).float().to(device)
    x = cnf(y, None, reverse=True).view(*y.size())
    return x


def plot_2D(samples, title):
    a = samples[:, 0]
    b = samples[:, 1]
    a_bins = np.linspace(-2, 3, 60)
    b_bins = np.linspace(-2, 2, 60)
    fig, ax = plt.subplots(figsize=(9, 4))
    plt.hist2d(a, b, bins=[a_bins, b_bins])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    plt.tight_layout()


def plot_2D_simple(samples, title):
    fig, ax = plt.subplots(figsize=(9, 4))
    plt.scatter(samples[:, 0], samples[:, 1], s=1)
    ax.set_title(title)
    plt.tight_layout()


def plot_1D(samples, title):
    a = samples[:, 0]
    a_bins = np.linspace(0, 1, 60)
    fig, ax = plt.subplots(figsize=(9, 4))
    plt.hist(a, bins=a_bins)
    ax.set_xlabel("X")
    ax.set_title(title)
    plt.tight_layout()


# main


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n, input_model, input_train_data = ParseArguments()
n = int(n)


samples_train = pd.read_pickle(input_train_data)
data_dim = samples_train.shape[1]

flow_model = torch.load(input_model)

samples_flow = sample_flow(flow_model, (n, data_dim), device)

if data_dim == 1:
    plot_1D(samples_train, title="Training data")
    plot_1D(samples_flow.detach().cpu().numpy(), title="Data sampled from flow")

if data_dim == 2:
    plot_2D(samples_train, title="Training data, v1")
    plot_2D(samples_flow.detach().cpu().numpy(), title="Data sampled from flow, v1")

    plot_2D_simple(samples_train, title="Training data, v2")
    plot_2D_simple(
        samples_flow.detach().cpu().numpy(), title="Data sampled from flow, v2"
    )

plt.show()
