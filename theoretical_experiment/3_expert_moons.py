#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import make_moons
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

from visual_tools import plot_HMM2, plot_Qs, plot_metric, plot_HMM3

PROJECT_PATH = Path(__file__).parent
# import sys
# sys.path.insert(1, PROJECT_PATH)
from torchHMM.utils.utils import total_variance_dist
from torchHMM.model.FlowHMM import FlowHMM, DISCRETIZATION_TECHNIQUES

LEARNING_ALGORITHMS = ["em", "cooc"]
T = 10000
np.random.seed(2023)
sns.set_style("white")

wandb_project_name = f"3_FlowHMM_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"

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
        params="ste",
        init_params="ste",
        optim_params=dict(max_epoch=50000, lr=0.1, weight_decay=0),
        n_iter=100,
    )

    model_._init(X_train_)
    model_.provide_nodes(X_train_, False)
    return model_


def list_grid_size():
    return [
        # 2**2,
        2**4,
        2**6,
        # 2**8
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
    ll = model.score(X_, np.array(X_.shape[0]))
    acc = accuracy(model_.predict(X_, np.array(X_.shape[0])), Z_)
    if Q_gt is not None:
        Q = Q_from_params(model_)
        kl = kl_divergence(Q, Q_gt)
        d_tv = total_variance_dist(Q, Q_gt)
    else:
        kl = None
        d_tv = None
    return {'kl': kl, 'll': ll, 'acc': acc, 'd_tv': d_tv, **info}


def init_params(X_expert, Z_expert):
    n_comp = Z_expert.max() + 1
    pi = np.ones(n_comp) / n_comp
    A = np.zeros((n_comp, n_comp))
    for i in range(Z_expert.shape[0] - 1):
        A[Z_expert[i], Z_expert[i+1]] += 1
    A = A / A.sum(axis=1).reshape(-1, 1)
    means = np.concatenate([X_expert[Z_expert == i, :].mean(axis=0).reshape(1, -1) for i in range(n_comp)])
    stds = np.concatenate([X_expert[Z_expert == i, :].std(axis=0).reshape(1, -1) for i in range(n_comp)])
    return pi, A, means, stds

results_path = f"{PROJECT_PATH}/theoretical_experiment/3_results_final"
Path(results_path).mkdir(exist_ok=True, parents=True)
grid_sizes = list_grid_size()


if __name__ == "__main__":
    X_train, Z_train = make_moons(T, random_state=2023, noise=0.05)
    X_test, Z_test = make_moons(T // 10, random_state=2022, noise=0.05)

    X_expert, Z_expert = make_moons(100, random_state=2021, noise=0.05)

    pi_init, A_init, means_init, covars_init = init_params(X_expert, Z_expert)

    results = list()

    for _ in tqdm(range(20)):  # the initialization differ in runs
        model = hmm.GaussianHMM(
            n_components=2,
            verbose=True,
            params="smct",
            init_params="",
            n_iter=100,
            covariance_type="full"
        )
        model.startprob_ = pi_init
        model.transmat_ = A_init
        model.means_ = means_init
        model.covars_ = np.concatenate([np.diag(x)[np.newaxis, :, :] for x in covars_init], axis=0)

        model.fit(X_train)

        results.append(
            score_model(model, X_test, Z_test, None, dict(discretization='none')))

    plot_HMM2(X_test, Z_test, model, path=f"{results_path}/3_gaussians_on_moons.png")

    for discretize_meth in DISCRETIZATION_TECHNIQUES:
        for n in grid_sizes:
            model = init_model(discretize_meth, X_train, n)

            for max_epoch, lr in itertools.product([100, 200],  [0.001, 0.003, 0.01])):

                for _ in tqdm(range(1)): # As we work with random methods, the initialization and  the discretization differ in runs
                    run = None
                    # run = wandb.init(
                    #    project=wandb_project_name,
                    #    name=f"ex_3_{discretize_meth}_{n}_{max_epoch}_{lr}",
                    #    notes="FlowHMM with co-occurrence-based learning schema logger",
                    #    dir=f'{PROJECT_PATH}/'
                    # )
                    # wandb.config = dict(max_epoch=max_epoch, lr=lr, weight_decay=0, disc=discretize_meth, n=n)
                    model = FlowHMM(
                        discretization_method=discretize_meth,
                        no_nodes=n,
                        n_components=2,
                        learning_alg="cooc",
                        verbose=True,
                        params="ste",
                        init_params="",
                        optim_params=dict(max_epoch=max_epoch, lr=lr, weight_decay=0, run=run),
                        n_iter=100,
                        optimizer="Adam",
                        means=means_init,
                        stds=covars_init
                    )

                    model.startprob_ = pi_init
                    model.transmat_ = A_init

                    model.fit(X_train)
                    # wandb.finish()

                    results.append(
                        score_model(model, X_test, Z_test, model._cooccurence(model.discretize(X_train, True)), dict(discretization=discretize_meth, n=n, max_epoch=max_epoch, lr=lr)))
                plot_HMM3(X_test, model, path=f"{results_path}/3_dist_on_moons_{discretize_meth}_{n}_{max_epoch}_{lr}.png")
                plot_Qs(Q_from_params(model), model._cooccurence(model.discretize(X_train, True)), f"{results_path}/3_Q_{discretize_meth}_{n}_{max_epoch}_{lr}.png")


                with open(
                    f"{results_path}/3_discretization.json",
                    "w",
                ) as f:
                    json.dump(results, f, indent=4)

    results = pd.DataFrame(results)
    for metric, title in zip(['d_tv', 'kl', 'acc', 'll'], ["Total variation distance", "KL divergence", 'State prediction accuracy', 'Loglikelihood']):
        plot_metric(results, metric, title, f"{results_path}/3_{metric}.png")
