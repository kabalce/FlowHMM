#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import make_moons
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import json
import datetime
from tqdm import tqdm
import itertools
from scipy.stats import multivariate_normal

from ssm.util import find_permutation
from pathlib import Path
from hmmlearn import hmm

from visual_tools import plot_HMM2, plot_Qs, plot_metric, plot_HMM3

PROJECT_PATH = Path(__file__).parent.parent
# import sys
# sys.path.insert(1, PROJECT_PATH)
from torchHMM.utils.utils import total_variance_dist
from torchHMM.model.FlowHMM import FlowHMM, DISCRETIZATION_TECHNIQUES

LEARNING_ALGORITHMS = ["em", "cooc"]
T = 10000
np.random.seed(2023)
sns.set_style("white")


def Q_from_params(model_):
    """
    Calculate Q from model parameters
    """
    if hasattr(model_, 'emissionprob_'):
        return Q_from_params_d(model_)

    S_ = model_.transmat_ * model_.startprob_[:, np.newaxis]
    distributions_ = [
        scipy.stats.multivariate_normal(model_.means_[i], model_.covars_[i])
        for i in range(model_.n_components)
    ]

    B_ = np.concatenate(
        [dist.pdf(model_.nodes.T).reshape(1, -1) for dist in distributions_],
        axis=0,
    )
    B_ = B_ / B_.sum(1)[:, np.newaxis]
    return B_.T @ S_ @ B_


def Q_from_params_d(model_):
    """
    Calculate Q from model parameters
    """
    S_ = model_.transmat_ * model_.startprob_[:, np.newaxis]
    B_ = model_.emissionprob_
    return B_.T @ S_ @ B_


def init_model(discretize_meth, X_train_, n):
    """
    Init DiscreteHMM with parameters from true model
    """
    model_ = FlowHMM(
        discretize_meth,
        n,
        learning_alg="cooc",
        verbose=True,
        params="mct",
        init_params="",
        optim_params=dict(max_epoch=50000, lr=0.1, weight_decay=0),
        n_iter=100,
    )

    model_._init(X_train_)
    model_.provide_nodes(X_train_, False)
    return model_


def list_grid_size(n_=3):
    return [
        10,
        int(np.ceil(0.5 * n_ * (1 + (2 * n_ - 1)**2))),
        50,
        int(np.ceil(np.sqrt(0.5 * n_ * (1 + (2 * n_ - 1)**2) * np.sqrt(T * n_ + 3**2)))),
        100,
        int(np.ceil(np.sqrt(T * n_ + n_**2))),
        250,
    ]


def kl_divergence(p_, q_):
    p = p_.reshape(-1) + 1e-10
    p /= p.sum()
    q = q_.reshape(-1) + 1e-10
    q /= q.sum()
    return np.sum(p * np.log2(p / q))


def accuracy(Z_hat, Z_):
    perm = find_permutation(np.concatenate([Z_hat, np.arange(max(Z_))]),
                            np.concatenate([Z_, np.arange(max(Z_))]))
    return (perm[Z_hat] == Z_).mean()

def score_model(model_, X_, Z_, Q_gt, info):
    ll = model.score(X_)
    acc = accuracy(model_.predict(X_), Z_)
    if Q_gt is not None:
        Q = Q_from_params(model_)
        kl = kl_divergence(Q, Q_gt)
        d_tv = total_variance_dist(Q, Q_gt)
    else:
        kl = None
        d_tv = None
    return {'kl': kl, 'll': ll, 'acc': acc, 'd_tv': d_tv, **info}

results_path = f"{PROJECT_PATH}/theoretical_experiment/2_results"
Path(results_path).mkdir(exist_ok=True, parents=True)
grid_sizes = list_grid_size()


if __name__ == "__main__":
    X_train, Z_train = make_moons(T, random_state=2023, noise=0.05)
    X_test, Z_test = make_moons(T // 10, random_state=2022, noise=0.05)

    results = list()

    for _ in tqdm(range(20)):  # the initialization differ in runs
        model = hmm.GaussianHMM(
            n_components=2,
            verbose=True,
            params="smct",
            init_params="smct",
            n_iter=100,
        )
        model.fit(X_train)

        results.append(
            score_model(model, X_test, Z_test, None, dict(discretization='none')))

    plot_HMM2(X_test, Z_test, model, path=f"{results_path}/2_gaussians_on_moons.png")

    for discretize_meth in DISCRETIZATION_TECHNIQUES:
        for n in grid_sizes:
            model = init_model(discretize_meth, X_train, n)

            for _ in tqdm(range(20)): # As we work with random methods, the initialization and  the discretization differ in runs
                model = FlowHMM(
                    discretization_method=discretize_meth,
                    no_nodes=n,
                    n_components=2,
                    learning_alg="cooc",
                    verbose=True,
                    params="mct",
                    init_params="mct",
                    optim_params=dict(max_epoch=50000, lr=0.01, weight_decay=0),
                    n_iter=100,
                )
                model.fit(X_train)

                results.append(
                    score_model(model, X_test, Z_test, model._cooccurence(model.discretize(X_train, True)), dict(discretization='discretize_meth', n=n)))
            plot_HMM3(X_test, Z_test, model, path= f"{results_path}/1_dist_on_moons_{discretize_meth}_{n}.png")
            plot_Qs(Q_from_params(model), model._cooccurence(model.discretize(X_train, True)), f"{results_path}/1_Q_{discretize_meth}_{n}.png")


    with open(
        f"{results_path}/2_flow_moons.json",
        "w",
    ) as f:
        json.dump(results, f)

    results = pd.DataFrame(results)
    for metric, title in zip(['d_tv', 'kl', 'acc', 'll'], ["Total variation distance", "KL divergence", 'State prediction accuracy', 'Loglikelihood']):
        plot_metric(results, metric, title, f"{results_path}/2_{metric}.png")
