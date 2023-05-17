import numpy as np


def normalize(embeddings, new_mean=0, new_std=1):
    mean, std = np.mean(embeddings), np.std(embeddings)
    embeddings += (new_mean - mean)
    embeddings *= (new_std / std)
    return embeddings


def flatten(xs):
    ys = []
    for x in xs:
        if isinstance(x, list):
            ys.extend(flatten(x))
        else:
            ys.append(x)

    return ys