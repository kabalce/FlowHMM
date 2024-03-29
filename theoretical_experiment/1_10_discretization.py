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
from sklearn.decomposition import PCA

PROJECT_PATH = Path(__file__).parent  # .parent
# import sys
# sys.path.insert(1, PROJECT_PATH)
from torchHMM.utils.utils import total_variance_dist
from torchHMM.model.GaussianHMM import DiscreteHMM, DISCRETIZATION_TECHNIQUES, HmmOptim

# from torchHMM.model.GaussianHMM2 import DiscreteHMM, DISCRETIZATION_TECHNIQUES, HmmOptim

LEARNING_ALGORITHMS = ["em", "cooc"]
T = 400000
np.random.seed(2023)
sns.set_style("white")

wandb_project_name = (
    f"1_GaussianHMM_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)
dim = 10
means = np.concatenate(
    [
        np.random.uniform(size=dim)[
            np.newaxis,
        ]
        * 8
        - 4
        for _ in range(3)
    ]
)


def LU(a):
    b = np.tril(a)
    return b.T @ b


covars = np.concatenate(
    [
        LU(np.random.uniform(size=(dim, dim)) + 0.1)[
            np.newaxis,
        ]
        for _ in range(3)
    ]
)


def init_true_model():
    true_model = hmm.GaussianHMM(n_components=3, covariance_type="full")

    true_model.startprob_ = np.array([0.6, 0.3, 0.1])
    true_model.transmat_ = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])

    true_model.means_ = means
    true_model.covars_ = covars

    true_model.n_features = dim

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
        scipy.stats.multivariate_normal(
            model_.means_[i], model_.covars_[i], allow_singular=True
        )
        for i in range(model_.n_components)
    ]

    B_ = np.concatenate(
        [dist.pdf(model_.nodes.T).reshape(1, -1) for dist in distributions_],
        axis=0,
    )
    B_ = B_ / np.maximum(B_.sum(1), 1e-12)[:, np.newaxis]
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
    return [2**4, 2**6, 2**8, 2**10]


def kl_divergence(p_, q_):
    p = p_.reshape(-1) + 1e-10
    p /= p.sum()
    q = q_.reshape(-1) + 1e-10
    q /= q.sum()
    return np.sum(p * np.log2(p / q))


def accuracy(Z_hat, Z_):
    perm = find_permutation(
        np.concatenate([Z_hat, np.arange(max(Z_))]),
        np.concatenate([Z_, np.arange(max(Z_))]),
    )
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
    return {"kl": kl, "ll": ll, "acc": acc, "d_tv": d_tv, **info}


results_path = f"{PROJECT_PATH}/1_{dim}_results_final"
Path(results_path).mkdir(exist_ok=True, parents=True)
grid_sizes = list_grid_size()


if __name__ == "__main__":
    true_model, _ = init_true_model()

    X_train, Z_train = true_model.sample(T)
    X_test, Z_test = true_model.sample(T // 10)

    pca = PCA(2).fit(X_train)

    X2 = pca.transform(X_train)

    x1, y1 = X2.min(axis=0) - 0.5
    x2, y2 = X2.max(axis=0) + 0.5

    XX, YY = np.meshgrid(np.linspace(x1, x2, 100), np.linspace(y1, y2, 100))
    data = np.column_stack((XX.ravel(), YY.ravel()))

    results = list()

    for _ in tqdm(range(1)):  # the initialization differ in runs
        model = hmm.GaussianHMM(
            n_components=3,
            verbose=True,
            params="smct",
            init_params="smct",
            n_iter=100,
        )
        model.fit(X_train)

        results.append(score_model(model, X_test, Z_test, None, dict()))
        print(results[-1])

    for discretize_meth in DISCRETIZATION_TECHNIQUES:
        for n in grid_sizes:
            model = init_model_with_params(discretize_meth, true_model, X_train, n)

            plot_HMM(  # Example plot
                X_train,
                model,
                discretize_meth,
                n,
                f"{results_path}/1_{dim}_nodes_{discretize_meth}_{n}.eps",
                data,
                pca,
                XX,
                YY,
                X2,
            )

            for max_epoch, lr in itertools.product([800], [0.001, 0.003, 0.01]):

                for _ in tqdm(
                    range(1)
                ):  # As we work with random methods, the initialization and  the discretization differ in runs
                    run = None
                    run = wandb.init(
                        project=wandb_project_name,
                        name=f"ex_1_{dim}_{discretize_meth}_{n}_{max_epoch}_{lr}",
                        notes="GaussianHMM with co-occurrence-based learning schema logger",
                        dir=f"{PROJECT_PATH}//wandb",
                    )
                    wandb.config = dict(
                        max_epoch=max_epoch,
                        lr=lr,
                        weight_decay=0,
                        disc=discretize_meth,
                        n=n,
                    )

                    model = DiscreteHMM(
                        discretization_method=discretize_meth,
                        no_nodes=n,
                        n_components=3,
                        learning_alg="cooc",
                        verbose=True,
                        params="mct",
                        init_params="mct",
                        optim_params=dict(
                            max_epoch=max_epoch, lr=lr, weight_decay=0, run=run
                        ),
                        n_iter=100,
                        covariance_type="full",
                        optimizer="Adam",
                    )
                    model.fit(X_train, early_stopping=True)
                    wandb.finish()

                    results.append(
                        score_model(
                            model,
                            X_test,
                            Z_test,
                            model._cooccurence(model.discretize(X_train, True)),
                            dict(
                                discretization=discretize_meth,
                                n=n,
                                max_epoch=max_epoch,
                                lr=lr,
                            ),
                        )
                    )
                    print(results[-1])
                # plot_Qs(Q_from_params(model), model._cooccurence(model.discretize(X_train, True)), f"{results_path}/1_Q_{discretize_meth}_{n}_{max_epoch}_{lr}.eps")

        with open(
            f"{results_path}/1_{dim}_discretization.json",
            "w",
        ) as f:
            json.dump(results, f, indent=4)

    results = pd.DataFrame(results)
    for metric, title in zip(
        ["d_tv", "kl", "acc", "ll"],
        [
            "Total variation distance",
            "KL divergence",
            "State prediction accuracy",
            "Loglikelihood",
        ],
    ):
        plot_metric(results, metric, title, f"{results_path}/1_{dim}_{metric}.eps")
