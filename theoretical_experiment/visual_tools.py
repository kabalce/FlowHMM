import matplotlib.pyplot as plt
import seaborn as sns
import urllib
from ssm.plots import gradient_cmap, white_to_color_cmap
import numpy as np
from scipy.stats import multivariate_normal
import torch


def profive_cmap():
    with urllib.request.urlopen('https://xkcd.com/color/rgb.txt') as f:
        colors = f.readlines()
    color_names = [str(c)[2:].split('\\t')[0] for c in colors[1:]]

    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors)
    return cmap, colors


cmap, colors = profive_cmap()


def plot_HMM(X, model, discretize_meth, n, path=None):
    """
    Plot emission distribution and nodes
    """
    norm1 = multivariate_normal(model.means_[0], model.covars_[0])
    norm2 = multivariate_normal(model.means_[1], model.covars_[1])
    norm3 = multivariate_normal(model.means_[2], model.covars_[2])
    norms = [norm1, norm2, norm3]

    x1, y1 = X.min(axis=0) - .5
    x2, y2 = X.max(axis=0) + .5

    XX, YY = np.meshgrid(np.linspace(x1, x2, 100), np.linspace(y1, y2, 100))
    data = np.column_stack((XX.ravel(), YY.ravel()))
    lls = np.concatenate([norm.pdf(data).reshape(-1, 1) for norm in norms], axis=1)

    plt.figure(figsize=(5, 5))
    for k in range(model.n_components):
        plt.contour(XX, YY, np.exp(lls[:, k]).reshape(XX.shape), cmap=white_to_color_cmap(colors[k]), levels=6)

    plt.scatter(model.nodes[0], model.nodes[1])
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.suptitle("True Distributions  and Nodes")
    plt.title(f"{discretize_meth} {n}")
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close()


def plot_Qs(Q_cooc, Q_true_model, path):
    """
    Plot comparison of 2 Q matrices
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(10, 10))
    vmin, vmax = min(Q_cooc.min(), Q_true_model.min()), max(Q_cooc.max(), Q_true_model.max())

    ax0.imshow(Q_cooc, vmin=vmin, vmax=vmax)
    ax0.set_title("Q from sample")

    ax1.imshow(Q_true_model, vmin=vmin, vmax=vmax)
    ax1.set_title("Q from parameters")

    plt.savefig(path)  # f"{PROJECT_PATH}/theoretical_experiment/plots/1_Q_{discretize_meth}_{n}_v2.png"
    plt.show()
    plt.close()


def plot_metric(results, metric, title, path, ylog=False):
    plot = sns.lineplot(results.loc[~results['n'].isna(), :], x='n', y=metric, hue='discretization', marker='o')
    h = results.loc[results['n'].isna(), metric].mean()
    if h is not None:
        plot.axhline(h)
    plt.title(title)
    plt.xlabel('number of unique discrete values')
    plt.legend(title="discretization\ntechnique")
    plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.legend(loc='center left', bbox_to_anchor=(1, 1))
    plt.savefig(path)
    plt.show()
    plt.close()


def plot_HMM2(X, Z, model, path=None):
    """
    Plot emission distribution
    """
    norm1 = multivariate_normal(model.means_[0], model.covars_[0])
    norm2 = multivariate_normal(model.means_[1], model.covars_[1])
    norm3 = multivariate_normal(model.means_[2], model.covars_[2])
    norms = [norm1, norm2, norm3]

    x1, y1 = X.min(axis=0) - .5
    x2, y2 = X.max(axis=0) + .5

    XX, YY = np.meshgrid(np.linspace(x1, x2, 100), np.linspace(y1, y2, 100))
    data = np.column_stack((XX.ravel(), YY.ravel()))
    # TODO: transformuj dane data siecią
    lls = np.concatenate([norm.pdf(data).reshape(-1, 1) for norm in norms], axis=1)

    plt.figure(figsize=(5, 5))
    for k in range(model.n_components):
        plt.scatter(X[Z == k, 0], X[Z == k, 1], color=colors[k], alpha=0.1)
        plt.contour(XX, YY, np.exp(lls[:, k]).reshape(XX.shape), cmap=white_to_color_cmap(colors[k]), levels=6)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(f"Gaussians on Moons")
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close()

def plot_HMM3(X, Z, model, path=None):
    """
    Plot emission distribution and nodes
    """
    norm = multivariate_normal(np.zeros(2), np.identity(2))

    x1, y1 = X.min(axis=0) - .5
    x2, y2 = X.max(axis=0) + .5

    XX, YY = np.meshgrid(np.linspace(x1, x2, 100), np.linspace(y1, y2, 100))
    data = np.column_stack((XX.ravel(), YY.ravel()))
    lls = norm.pdf(data).reshape(-1, 1)

    plt.figure(figsize=(5, 5))
    for k in range(model.n_components):
        plt.scatter(X[Z == k, 0], X[Z == k, 1], color=colors[k], alpha=0.1)
        data_NF = model.model.NFs[k](torch.tensor(data).to(model.model.device)).cpu().detach().numpy()
        plt.contour(data_NF[:, 0], data_NF[:, 1], np.exp(lls[:, k]).reshape(XX.shape), cmap=white_to_color_cmap(colors[k]), levels=6)

    plt.scatter(model.nodes[0], model.nodes[1])
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.suptitle("True Distributions and Nodes")
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close()
