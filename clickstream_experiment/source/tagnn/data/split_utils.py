import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Set, Dict


def create_k_core(interactions_pdf: pd.DataFrame, users_k: int = 5, items_k: int = 5) -> pd.DataFrame:
    """
    Iteratively removes every user and item with interactions count below threshold.

    Args:
        interactions_pdf: Pandas dataframe containing all the interactions.
        users_k: Minimum number of interactions required for users.
        items_k: Minimum number of interactions required for items.

    Returns:
        A pandas dataframe containing interactions where each user and item meets the criteria.
    """
    while np.any(interactions_pdf['asin'].value_counts() < items_k) or np.any(
            interactions_pdf['reviewerID'].value_counts() < users_k):
        items_to_keep = interactions_pdf['asin'].value_counts().where(lambda x: x >= items_k).dropna().index
        users_to_keep = interactions_pdf['reviewerID'].value_counts().where(lambda x: x >= users_k).dropna().index
        interactions_pdf = interactions_pdf[
            interactions_pdf['asin'].isin(items_to_keep) & interactions_pdf['reviewerID'].isin(users_to_keep)
            ]

    return interactions_pdf


def get_train_core(interactions_pdf: pd.DataFrame, users_k: int = 5, items_k: int = 5) -> Set[np.int64]:
    """
    Finds a small subset of interactions where each user and item has the required number of interactions.

    As opposed to create_k_core, this function doesn't keep all the interactions for chosen users/ items.
    Instead, it constructs a set of interactions that should not be included in the test set, so that for every
    user and item in the test set, a sufficient number of interactions remains in the training set.
    Args:
        interactions_pdf: Pandas dataframe containing all the interactions.
        users_k: Minimum number of interactions required for users.
        items_k: Minimum number of interactions required for items.

    Returns:
        Indices of interactions where each user and item meets the criteria.
    """
    train_core = []
    for _, pdf in tqdm(interactions_pdf.groupby('reviewerID')):
        train_core.extend(np.random.choice(pdf.index, users_k, replace=False))

    for _, pdf in tqdm(interactions_pdf.groupby('asin')):
        train_core.extend(np.random.choice(pdf.index, items_k, replace=False))

    return set(train_core)


def create_subsets(
        positive_interactions_pdf: pd.DataFrame,
        negative_interactions_pdf: pd.DataFrame,
        test_interactions: Set[np.int64],
) -> Dict[str, pd.DataFrame]:
    """
    Constructs train (positive and negative) and test sets based on the test interactions set.

    Args:
        positive_interactions_pdf: Pandas dataframe containing positive interactions from which to construct training
            and test sets.
        negative_interactions_pdf: Pandas dataframe containing negative interactions to compliment the training set.
        test_interactions: A set of interactions that should be included in the test set; all of those interactions
            should be present in positive_interactions_pdf.

    Returns:
        A dict mapping names to corresponding subsets.

    """
    train_pdf = positive_interactions_pdf[~positive_interactions_pdf.index.isin(test_interactions)]
    test_pdf = positive_interactions_pdf[positive_interactions_pdf.index.isin(test_interactions)]
    test_pdf = test_pdf[
        test_pdf['reviewerID'].isin(train_pdf['reviewerID'].unique()) &
        test_pdf['asin'].isin(train_pdf['asin'].unique())
        ]
    negative_pdf = negative_interactions_pdf[
        negative_interactions_pdf['reviewerID'].isin(train_pdf['reviewerID'].unique()) &
        negative_interactions_pdf['asin'].isin(train_pdf['asin'].unique())
        ]
    return {
        'train': train_pdf,
        'test': test_pdf,
        'negative_train': negative_pdf,
    }


def save_subsets(path: Path, dataset_name: str, sets: Dict[str, pd.DataFrame], file_format: str = 'parquet') -> None:
    """
    Saves the dataset after splittings.

    Each subset is saved to path / dataset_name / subset_name.

    Args:
        path: Path in which to save the dataset.
        dataset_name: Name of the dataset.
        sets: A dict mapping names to corresponding subsets.
        file_format: Format in which to save files.
    """
    if not (path / dataset_name).exists():
        os.mkdir(path / dataset_name)
    for subset_name, subset in sets.items():
        if file_format == 'parquet':
            subset.to_parquet(path / dataset_name / f'{subset_name}.parquet')
        elif file_format == 'csv':
            subset.to_csv(path / dataset_name / f'{subset_name}.csv', index=False)
        else:
            raise ValueError('Format not supported.')


def choose_test_users(
        test_interactions_pdf: pd.DataFrame,
        num_users: int = 10000,
        balanced: bool = True,
        cond=lambda x: x >= 5
) -> Set[str]:
    """
    Randomly chooses the users to be included in the test set.

    Args:
        test_interactions_pdf: A pandas dataframe containing interactions from which test set should be built.
        num_users: Number of users to sample.
        balanced: If true, users with higher interaction counts are more likely to be chosen (as the majority of
            users have low interaction counts).
        cond: A condition for pd.Series.where for filtering the user interaction counts.

    Returns:
        A set of users that should be included in the test set.
    """
    user_counts = test_interactions_pdf['reviewerID'].value_counts().where(cond).dropna()
    probabilities = user_counts.values / np.sum(user_counts.values) if balanced else None

    return set(np.random.choice(user_counts.index, num_users, replace=False, p=probabilities))


