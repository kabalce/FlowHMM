import numpy as np


def hit_rate_from_ranks(ranks, k=10):
    ranks = np.array(ranks)
    return np.sum(ranks < k) / len(ranks)


def ndcg_from_ranks(ranks, k=10):
    good_ranks = ranks[ranks < k]
    return np.sum(1 / np.log2(good_ranks + 2)) / len(ranks)