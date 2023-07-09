#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown, Latex
import json
import urllib
import datetime
from tqdm import tqdm
import itertools
from scipy.stats import multivariate_normal

from ssm.util import find_permutation
from ssm.plots import gradient_cmap, white_to_color_cmap
from pathlib import Path
from hmmlearn import hmm
import sys
PROJECT_PATH = Path(__file__).parent.parent
sys.path.insert(1, PROJECT_PATH)
from torchHMM.utils.utils import total_variance_dist
from torchHMM.model.discretized_HMM import DiscreteHMM, DISCRETIZATION_TECHNIQUES, HmmOptim

LEARNING_ALGORITHMS = ["em", "cooc"]
T = 10000
np.random.seed(2023)
sns.set_style("white")


def profive_cmap():
    with urllib.request.urlopen('https://xkcd.com/color/rgb.txt') as f:
        colors = f.readlines()
    color_names = [str(c)[2:].split('\\t')[0] for c in colors[1:]]

    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors)
    return cmap, colors


def init_true_model():
    true_model = hmm.GaussianHMM(n_components=3, covariance_type="full")

    true_model.startprob_ = np.array([0.6, 0.3, 0.1])
    true_model.transmat_ = np.array([[0.7, 0.2, 0.1],
                                [0.3, 0.5, 0.2],
                                [0.3, 0.3, 0.4]])

    true_model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [4.0, 3.0]])
    true_model.covars_ = np.array([[[1, -.5], [-.5, 1.2]], [[.6, -.5], [-.5, 1.2]], [[1.5, .5], [.5, 2.2]]]) * .8

    true_model.n_features = 2

    norm1 = multivariate_normal(true_model.means_[0], true_model.covars_[0])
    norm2 = multivariate_normal(true_model.means_[1], true_model.covars_[1])
    norm3 = multivariate_normal(true_model.means_[2], true_model.covars_[2])
    norms = [norm1, norm2, norm3]

    return true_model, norms


# Wybór metryk i sposób prezentacji:
# 
# 1. Dla 25, 100, 400 węzłów:
#     - wykres: rozkłady ze stanów a węzły
#     - wykres: zgodność macierzy sąsiedztwa
#     - powtórz 100 razy:
#         - KL: zgodność macierzy sąsiedztwa
#         - d_{tv}: zgodność macierzy sąsiedztwa
#     - wykres: metryki z przedziałami ufności

# In[9]:


def plot_Q_from_model(model_):
    S_ = model_.transmat_ * model_.startprob_[:, np.newaxis]
    distributions_ = [
            scipy.stats.multivariate_normal(model_.means_[i], model_.covars_[i])
            for i in range(model_.n_components)
        ]

    B_ = np.concatenate(
            [
                dist.pdf(model_.nodes.T).reshape(
                    1, -1
                )
                for dist in distributions_
            ],
            axis=0,
        )
    B_ = B_ / B_.sum(1)[:, np.newaxis]
    return B_.T @ S_ @ B_


def init_model(discretize_meth, true_model, n):
    model = DiscreteHMM(discretize_meth, n, n_components=3, learning_alg='cooc', verbose=True, params="mct", init_params="",
                        optim_params=dict(max_epoch=50000, lr=0.1, weight_decay=0), n_iter=100)

    model.startprob_ = true_model.startprob_
    model.transmat_ = true_model.transmat_
    model.means_ = true_model.means_
    model.covars_ = true_model.covars_
    return model


def plot_true_HMM(X, model, path=None, colors=[]):

    x1, y1 = X.min(axis=0) * 1.1
    x2, y2 = X.max(axis=0) * 1.1

    XX, YY = np.meshgrid(np.linspace(x1, x2, 100), np.linspace(y1, y2, 100))
    data = np.column_stack((XX.ravel(), YY.ravel()))
    lls = np.concatenate([norm.pdf(data).reshape(-1, 1) for norm in norms], axis=1)

    plt.figure(figsize=(5, 5))
    for k in range(model.n_components):
        plt.contour(XX, YY, np.exp(lls[:,k]).reshape(XX.shape), cmap=white_to_color_cmap(colors[k]), levels=8)

    plt.scatter(model.nodes[0], model.nodes[1])
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Observation Distributions")
    if path is not None:
        plt.savefig(path)
    plt.show()

def kl_divergence(p_, q_):
    p = p_.reshape(-1) + 1e-6
    p /= p.sum()
    q = q_.reshape(-1) + 1e-6
    q /= q.sum()
    return np.sum(p * np.log2(p / q))


# Selecting number of nodes:
# 
# - NNMF-HMM: $\sqrt{TN + N^2}$
# - minimal: $0.5 (N(1 + \sqrt{5}))$
# - try also geometric mean of above

# In[10]:

def plot_Q_cooc(Q_cooc, discretize_meth, n):
    fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(10, 10))
    vmin, vmax = Q_cooc.min(), Q_cooc.max()
    ax0.imshow(Q_cooc, vmin=vmin, vmax=vmax)
    ax0.set_title("Q from sample")
    Q_true_model = plot_Q_from_model(model)
    ax1.imshow(Q_true_model, vmin=vmin, vmax=vmax)
    ax1.set_title("Q from parameters")
    plt.savefig(f"{PROJECT_PATH}/theoretical_experiment/plots/1_Q_{discretize_meth}_{n}_v2.png")
    plt.show()
    return Q_true_model



# In[24]:

def plot_metric(results, metric, title, train):
    sns.lineplot(results, x='n', y=metric, hue='disc', marker='o')
    plt.title(title)
    plt.xlabel('number of unique discrete values')
    plt.legend(title="discretization\ntechnique")
    plt.xscale('log')
    plt.savefig(f"{PROJECT_PATH}/theoretical_experiment/plots/1_{metric}_trained={train}.png")
    plt.show()



if __name__ == "__main__":
    cmap, colors = profive_cmap()
    true_model, norms = init_true_model()

    X_train, Z_train = true_model.sample(T)
    X_test, Z_test = true_model.sample(T // 10)

    ll_file = open(f"{PROJECT_PATH}/theoretical_experiment/1_discretization_ll_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}.txt", "w")

    print("Standard model score: ", true_model.score(X_test))
    ll_file.write(f"Standard model score: {true_model.score(X_test)}")

    results = list()
    Path(f"{PROJECT_PATH}/theoretical_experiment/plots").mkdir(exist_ok=True, parents=True)
    for discretize_meth in DISCRETIZATION_TECHNIQUES:
        for n in [int(np.ceil(.5 * 3 * (1 + 5 ** 2))), int(np.ceil(np.sqrt(.5 * 3 * (1 + 5 ** 2) * np.sqrt(T * 3 + 3 ** 2)))), int(np.ceil(np.sqrt(T * 3 + 3 ** 2)))]:
            model = init_model(discretize_meth, true_model, n)
            model._init(X_train)
            Xd = model.discretize(X_test, True)

            # RQ1: How much do we disturb the distribution?
            print(f"{discretize_meth} {n}\n\tDiscretized model score: ", true_model.score(Xd))
            ll_file.write(f"{discretize_meth} {n}\n\tDiscretized model score: {true_model.score(Xd)}")


            # Q_cooc = model._cooccurence(Xd)
            # Q_true_model = plot_Q_cooc(Q_cooc, discretize_meth, n)
            plot_true_HMM(X_train, model,
                          f"{PROJECT_PATH}/theoretical_experiment/plots/1_nodes_{discretize_meth}_{n}_v2.png", colors)

            for _ in tqdm(range(20)):
                X, Z = true_model.sample(T)
                model = DiscreteHMM(discretize_meth, n, n_components=3, learning_alg='cooc', verbose=True, params="", init_params="",
                                    optim_params=dict(max_epoch=50000, lr=0.01, weight_decay=0), n_iter=100)
                # model._init(X)
                model.fit(X)
                Xd = model.discretize(X, True)

                Q_cooc = model._cooccurence(Xd)
                Q_true_model = plot_Q_from_model(model)

                kl = kl_divergence(Q_cooc, Q_true_model)
                dtv = total_variance_dist(Q_cooc, Q_true_model)
                results.append({'KL': kl, 'd_tv': dtv, 'disc': discretize_meth, 'n': n})

    ll_file.close()
    with open(
            f"{PROJECT_PATH}/theoretical_experiment/1_discretization_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}.json",
            'w') as f:
        json.dump(results, f)

    results = pd.DataFrame(results)
    # results0 = pd.DataFrame(results)
    plot_metric(results0, "d_tv", 'Total variation distance')
    plot_metric(results, "d_tv", 'Total variation distance', train=True)
    plot_metric(results0, "KL", 'KL divergence')
    plot_metric(results, "KL", 'KL divergence', train=True)
