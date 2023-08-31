import random

import numpy as np
import pandas as pd
from scipy import spatial
from tqdm.auto import tqdm

DATA_PATH = "/pio/scratch/1/recommender_systems"


def prepare_abx_tests(
    categories_pdf,
    test_set_size=10000,
    category_column_name="category_1",
    item_column_name="asin",
    weighted_sample=True,
    min_counts=0,
    items_to_choose_from=None,
    seed=None,
):

    random.seed(seed)

    counts = categories_pdf[category_column_name].value_counts()
    counts = counts[counts > min_counts]
    categories_grouped = categories_pdf.groupby(category_column_name)

    categories_dict = {}
    for category, items in categories_grouped:
        if items_to_choose_from is not None:
            categories_dict[category] = items.loc[
                items[item_column_name].isin(items_to_choose_from)
            ]
        else:
            categories_dict[category] = items

    test_set = []
    for _ in tqdm(range(test_set_size)):

        # choosing categories for ABX
        if weighted_sample:
            category_sample = counts.sample(2, weights=counts)
        else:
            category_sample = counts.sample(2)

        # choosing which category to treat as positive & negative
        positive = category_sample.index[0]
        negative = category_sample.index[1]
        if random.random() < 0.5:
            positive, negative = negative, positive

        # choosing items for ABX from positive & nega
        positive_items = categories_dict[positive].sample(2)[item_column_name].values
        negative_item = categories_dict[negative].sample(1)[item_column_name].values

        # appending record
        line = {
            "A": positive_items[0],
            "B": negative_item[0],
            "X": positive_items[1],
            "category_AX": positive,
            "category_B": negative,
        }
        test_set.append(line)

    return pd.DataFrame(test_set)


def calculate_abx_score(items_embeddings, abx_tests_pdf):
    lines = len(abx_tests_pdf)

    A = items_embeddings[abx_tests_pdf["A"]]
    B = items_embeddings[abx_tests_pdf["B"]]
    X = items_embeddings[abx_tests_pdf["X"]]

    dist_A = ((A - X) ** 2).sum(axis=1)
    dist_B = ((B - X) ** 2).sum(axis=1)

    cos_dist_A = np.zeros(lines)
    cos_dist_B = np.zeros(lines)

    for i in range(lines):
        cos_dist_A[i] = spatial.distance.cosine(A[i, :], X[i, :])
        cos_dist_B[i] = spatial.distance.cosine(B[i, :], X[i, :])

    return {
        "abx_euclidean": np.mean(dist_A < dist_B),
        "abx_cosine": np.mean(cos_dist_A < cos_dist_B),
    }
