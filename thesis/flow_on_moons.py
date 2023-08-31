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
import torch
from hmmlearn import hmm

from theoretical_experiment.visual_tools import (
    plot_HMM2,
    plot_Qs,
    plot_metric,
    plot_HMM3,
)

PROJECT_PATH = Path(__file__).parent.parent
# import sys
# sys.path.insert(1, PROJECT_PATH)
from torchHMM.utils.utils import total_variance_dist
from torchHMM.model.FlowHMM import FlowHMM, DISCRETIZATION_TECHNIQUES

LEARNING_ALGORITHMS = ["em", "cooc"]
T = 10000
np.random.seed(2023)
sns.set_style("white")

wandb_project_name = f"2_FlowHMM_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"


def Q_from_params(model_):
    """
    Calculate Q from model parameters
    """
    if hasattr(model_, "emissionprob_"):
        return (
            model_.model(torch.tensor(model_.nodes.T).float())[0].cpu().detach().numpy()
        )

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
        params="stmc",
        init_params="stmc",
        optim_params=dict(max_epoch=50000, lr=0.1, weight_decay=0),
        n_iter=100,
    )

    model_._init(X_train_)
    model_.provide_nodes(X_train_, False)
    return model_


def list_grid_size():
    return [
        # 2**2,
        # 2**4,
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
    perm = find_permutation(
        np.concatenate([Z_hat, np.arange(max(Z_))]),
        np.concatenate([Z_, np.arange(max(Z_))]),
    )
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
    return {"kl": kl, "ll": ll, "acc": acc, "d_tv": d_tv, **info}


results_path = f"{PROJECT_PATH}/thesis/runs_64"
Path(results_path).mkdir(exist_ok=True, parents=True)
grid_sizes = list_grid_size()


if __name__ == "__main__":
    X_train, Z_train = make_moons(T, random_state=2023, noise=0.05)
    X_test, Z_test = make_moons(T // 10, random_state=2022, noise=0.05)

    results = list()

    for discretize_meth in DISCRETIZATION_TECHNIQUES:
        for n in grid_sizes[-1:]:
            model = init_model(discretize_meth, X_train, n)

            for max_epoch, lr, lambda_ in itertools.product([1000], [0.01], [0]):

                for _ in tqdm(
                    range(1)
                ):  # As we work with random methods, the initialization and  the discretization differ in runs
                    run = None
                    # run = wandb.init(
                    #    project=wandb_project_name,
                    #    name=f"ex_2_{discretize_meth}_{n}_{max_epoch}_{lr}",
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
                        params="stmc",
                        init_params="stmc",
                        optim_params=dict(max_epoch=10, lr=lr, weight_decay=0, run=run),
                        n_iter=100,
                        optimizer="Adam",
                    )

                    Xd = model.discretize(X_train, False)
                    lengths_d = None
                    super(type(model), model)._init(X_train)
                    model._init(X_train, None)

                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    model.model.to(device)
                    model.model.device = device
                    model.model.means = model.model.means.to(
                        device
                    )  # could be nicer...
                    model.model.stds = model.model.stds.to(device)

                    cooc_matrix = (
                        torch.tensor(model._cooccurence(Xd, None))
                        .to(device)
                        .requires_grad_(False)
                    )
                    run = (
                        model.optim_params.pop("run")
                        if "run" in model.optim_params.keys()
                        else None
                    )
                    optimizer = model.optimizer(
                        model.model.parameters(), **model.optim_params
                    )
                    nodes_tensor = (
                        torch.tensor(model.nodes.copy().T)
                        .float()
                        .to(device)
                        .requires_grad_(False)
                    )

                    scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer, gamma=0.9
                    )

                    for i in range(20):
                        plot_HMM3(
                            X_test,
                            model,
                            path=f"{results_path}/flow_on_moons_{i}_64_penalty={lambda_}_{discretize_meth}.png",
                        )
                        # plot_Qs(Q_from_params(model), model._cooccurence(model.discretize(X_train, False)),
                        #         f"{results_path}/Q_moons_{i}_64_penalty={lambda_}_{discretize_meth}.png")
                        results.append(
                            score_model(
                                model,
                                X_test,
                                Z_test,
                                model._cooccurence(model.discretize(X_train, False)),
                                dict(i=i * 20, penalty=lambda_),
                            )
                        )
                        for _ in range(model.max_epoch):
                            optimizer.zero_grad()
                            Q_hat, probs_sums = model.model(nodes_tensor)
                            loss = (
                                torch.nn.KLDivLoss(reduction="sum")(
                                    torch.log(Q_hat), cooc_matrix
                                )
                                - lambda_ * probs_sums.sum() / model.n_components
                            )
                            loss.backward()
                            optimizer.step()
                            if i % 10 == 0:  # TODO: think of it...
                                (
                                    _,
                                    model.transmat_,
                                    model.startprob_,
                                ) = model.model.get_model_params(nodes_tensor)

                                if run is not None:
                                    run.log(
                                        {
                                            "score": model.score(X_train, None),
                                            "loss": loss.cpu().detach(),
                                        }
                                    )
                                else:
                                    print(
                                        {
                                            "score": model.score(X_train, None),
                                            "loss": loss.cpu().detach(),
                                        }
                                    )

                            elif i % 100 == 99:  # TODO: select properly
                                (
                                    _,
                                    model.transmat_,
                                    model.startprob_,
                                ) = model.model.get_model_params(nodes_tensor)

                                scheduler.step()
                                if X_train is not None:
                                    score = model.score(Xd.reshape(-1, 1), None)
                                    model.monitor_.report(score)
                                    if (
                                        False and model.monitor_.converged
                                    ):  # TODO: monitor convergence from torch training
                                        break

                    i += 1

                    plot_HMM3(
                        X_test,
                        model,
                        path=f"{results_path}/flow_on_moons_{i}_64_penalty={lambda_}_{discretize_meth}.png",
                    )
                    # plot_Qs(Q_from_params(model), model._cooccurence(model.discretize(X_train, True)),
                    #         f"{results_path}/Q_moons_{i}_64_penalty={lambda_}.png")
                    results.append(
                        score_model(
                            model,
                            X_test,
                            Z_test,
                            model._cooccurence(model.discretize(X_train, True)),
                            dict(i=i * 20, penalty=lambda_),
                        )
                    )

                    # wandb.finish()

                with open(
                    f"{results_path}/single_run_64.json",
                    "w",
                ) as f:
                    json.dump(results, f, indent=4)
