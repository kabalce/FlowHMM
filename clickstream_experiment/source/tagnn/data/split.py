import numpy as np
from tqdm import tqdm

from src.data.split_utils import create_k_core, get_train_core, choose_test_users, create_subsets


def core_train_test_split(
        interactions_pdf,
        users_k=5,
        items_k=5,
        num_samples=5,
        num_users=10000,
        min_interactions=5,
        balanced=True
):
    negative_interactions_pdf = interactions_pdf[interactions_pdf['overall'] < 4]
    interactions_pdf = interactions_pdf[interactions_pdf['overall'] >= 4]

    interactions_pdf = create_k_core(interactions_pdf, users_k, items_k)
    train_core = get_train_core(interactions_pdf, users_k, items_k)
    non_core_interactions_pdf = interactions_pdf[~interactions_pdf.index.isin(train_core)]

    test_users = choose_test_users(
        non_core_interactions_pdf,
        cond=lambda x: x >= min_interactions,
        num_users=num_users,
        balanced=balanced
    )

    test_interactions = set()
    for user, pdf in tqdm(non_core_interactions_pdf.groupby('reviewerID')):
        if user in test_users:
            test_interactions |= set(np.random.choice(pdf.index, num_samples, replace=False))

    return create_subsets(interactions_pdf, negative_interactions_pdf, test_interactions)


def frac_train_test_split(interactions_pdf, frac=0.3, num_users=10000, min_interactions=5, balanced=True):
    negative_interactions_pdf = interactions_pdf[interactions_pdf['overall'] < 4]
    interactions_pdf = interactions_pdf[interactions_pdf['overall'] >= 4]

    test_users = choose_test_users(
        interactions_pdf,
        cond=lambda x: np.floor((1-frac)*x) >= min_interactions,
        num_users=num_users,
        balanced=balanced
    )

    test_interactions = set()
    for user, pdf in tqdm(interactions_pdf.groupby('reviewerID')):
        if user in test_users:
            test_interactions |= set(np.random.choice(pdf.index, int(np.ceil(frac*len(pdf))), replace=False))

    return create_subsets(interactions_pdf, negative_interactions_pdf, test_interactions)


def const_train_test_split(interactions_pdf, num_samples=5, num_users=10000, min_interactions=5, balanced=True):
    negative_interactions_pdf = interactions_pdf[interactions_pdf['overall'] < 4]
    interactions_pdf = interactions_pdf[interactions_pdf['overall'] >= 4]

    test_users = choose_test_users(
        interactions_pdf,
        cond=lambda x: x >= min_interactions,
        num_users=num_users,
        balanced=balanced
    )

    test_interactions = set()
    for user, pdf in tqdm(interactions_pdf.groupby('reviewerID')):
        if user in test_users:
            test_interactions |= set(np.random.choice(pdf.index, num_samples, replace=False))

    return create_subsets(interactions_pdf, negative_interactions_pdf, test_interactions)


def time_train_test_split(interactions_pdf, frac=0.1):
    interactions_pdf = interactions_pdf.sort_values('unixReviewTime')
    split_time = interactions_pdf.iloc[int((1-frac)*len(interactions_pdf))]['unixReviewTime']

    train_pdf = interactions_pdf[interactions_pdf['unixReviewTime'] < split_time]
    test_pdf = interactions_pdf[interactions_pdf['unixReviewTime'] >= split_time]

    train_pos_pdf = train_pdf[train_pdf['overall'] >= 4]
    test_clean_pdf = test_pdf[
        test_pdf['reviewerID'].isin(train_pos_pdf['reviewerID'].unique())
        & test_pdf['asin'].isin(train_pos_pdf['asin'].unique())
    ]

    return {
        'train': train_pos_pdf,
        'train_negative': train_pdf[train_pdf['overall'] < 4],
        'test': test_pdf[test_pdf['overall'] >= 4],
        'test_negative': test_pdf[test_pdf['overall'] < 4],
        'test_clean': test_clean_pdf[test_clean_pdf['overall'] >= 4],
        'test_clean_negative': test_clean_pdf[test_clean_pdf['overall'] < 4],
    }


def leave_one_out(interactions_pdf, item_col='itemID', user_col='userID', timestamp_col='timestamp'):
    test_pdf = interactions_pdf.sort_values(timestamp_col).drop_duplicates(user_col, keep='last')
    train_pdf = interactions_pdf[~interactions_pdf.index.isin(test_pdf.index)]
    test_pdf = test_pdf[
        (test_pdf[item_col].isin(train_pdf[item_col].unique()))
        & (test_pdf[user_col].isin(train_pdf[user_col].unique()))
        ]
    return train_pdf, test_pdf
