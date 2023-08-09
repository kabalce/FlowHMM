#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import json
import datetime
import wandb
from tqdm import tqdm
import itertools
from scipy.stats import multivariate_normal

from ssm.util import find_permutation
from pathlib import Path
from hmmlearn import hmm

from theoretical_experiment.visual_tools import plot_HMM, plot_Qs, plot_metric

PROJECT_PATH = Path(__file__).parent # .parent
# import sys
# sys.path.insert(1, PROJECT_PATH)
from torchHMM.utils.utils import total_variance_dist
from torchHMM.model.GaussianHMM import DiscreteHMM, DISCRETIZATION_TECHNIQUES, HmmOptim

LEARNING_ALGORITHMS = ["em", "cooc"]
T = 10000
np.random.seed(2023)
sns.set_style("white")


def init_true_model():
    true_model = hmm.GaussianHMM(n_components=3, covariance_type="full")

    true_model.startprob_ = np.array([0.6, 0.3, 0.1])
    true_model.transmat_ = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])

    true_model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [4.0, 3.0]])
    true_model.covars_ = (
        np.array(
            [
                [[1, -0.5], [-0.5, 1.2]],
                [[0.6, -0.5], [-0.5, 1.2]],
                [[1.5, 0.5], [0.5, 2.2]],
            ]
        )
        * 0.8
    )

    true_model.n_features = 2

    norm1 = multivariate_normal(true_model.means_[0], true_model.covars_[0])
    norm2 = multivariate_normal(true_model.means_[1], true_model.covars_[1])
    norm3 = multivariate_normal(true_model.means_[2], true_model.covars_[2])
    norms = [norm1, norm2, norm3]

    return true_model, norms


def Q_from_params(model_):
    """
    Calculate Q from model parameters
    """
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


def init_model_with_params(discretize_meth, true_model_, X_train_, n):
    """
    Init DiscreteHMM with parameters from true model
    """
    model_ = DiscreteHMM(
        discretize_meth,
        n,
        n_components=3,
        learning_alg="cooc",
        verbose=True,
        params="mct",
        init_params="",
        optim_params=dict(max_epoch=50000, lr=0.1, weight_decay=0),
        n_iter=100,
    )

    model_.startprob_ = true_model_.startprob_
    model_.transmat_ = true_model_.transmat_
    model_.means_ = true_model_.means_
    model_.covars_ = true_model_.covars_

    model_._init(X_train_)
    model_.provide_nodes(X_train_, False)
    return model_


def list_grid_size():
    return [
        2**2,
        2**4,
        2**6
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

results_path = f"{PROJECT_PATH}/theoretical_experiment/1_results_final"
Path(results_path).mkdir(exist_ok=True, parents=True)
grid_sizes = list_grid_size()


if __name__ == "__main__":
    true_model, _ = init_true_model()

    X_train, Z_train = true_model.sample(T)
    X_test, Z_test = true_model.sample(T // 10)

    results = list()

    for _ in tqdm(range(20)):  # the initialization differ in runs
        model = hmm.GaussianHMM(
            n_components=3,
            verbose=True,
            params="smct",
            init_params="smct",
            n_iter=100,
        )
        model.fit(X_train)

        results.append(
            score_model(model, X_test, Z_test, None, dict()))

    for discretize_meth in DISCRETIZATION_TECHNIQUES:
        for n in grid_sizes:
            model = init_model_with_params(discretize_meth, true_model, X_train, n)

            plot_HMM(  # Example plot
                X_train,
                model,
                discretize_meth,
                n,
                f"{results_path}/1_nodes_{discretize_meth}_{n}.png",
            )

            for max_epoch, lr in itertools.product([1000, 10000, 20000],  [0.01, 0.03, 0.1]):

                for _ in tqdm(range(20)): # As we work with random methods, the initialization and  the discretization differ in runs
                    run = wandb.init(
                        project="GaussianHMM",
                        name=f"ex_1_{discretize_meth}_{n}_{max_epoch}_{lr}",
                        notes="GaussianHMM with co-occurrence-based learning schema logger"
                    )
                    wandb.config = dict(max_epoch=max_epoch, lr=lr, weight_decay=0, disc=discretize_meth, n=n)

                    model = DiscreteHMM(
                        discretization_method=discretize_meth,
                        no_nodes=n,
                        n_components=3,
                        learning_alg="cooc",
                        verbose=True,
                        params="mct",
                        init_params="mct",
                        optim_params=dict(max_epoch=max_epoch, lr=lr, weight_decay=0, run=run),
                        n_iter=100,
                    )
                    model.fit(X_train)
                    wandb.finish()

                    results.append(
                        score_model(model, X_test, Z_test, model._cooccurence(model.discretize(X_train, True)), dict(discretization=discretize_meth, n=n, max_epoch=max_epoch, lr=lr)))

                plot_Qs(Q_from_params(model), model._cooccurence(model.discretize(X_train, True)), f"{results_path}/1_Q_{discretize_meth}_{n}_{max_epoch}_{lr}.png")


    with open(
        f"{results_path}/1_discretization.json",
        "w",
    ) as f:
        json.dump(results, f, indent=4)

    results = pd.DataFrame(results)
    for metric, title in zip(['d_tv', 'kl', 'acc', 'll'], ["Total variation distance", "KL divergence", 'State prediction accuracy', 'Loglikelihood']):
        plot_metric(results, metric, title, f"{results_path}/1_{metric}.png")
