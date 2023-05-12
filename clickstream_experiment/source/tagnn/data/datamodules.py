import os
from ast import literal_eval
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from recommenders.datasets import movielens
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.data.abx import prepare_abx_tests
from src.data.datasets import InteractionsDataset
from src.data.split import leave_one_out


def read_test(file_path):
    users = []
    items = []
    ratings = []
    with open(file_path) as in_file:
        for line in in_file:
            line = [literal_eval(elem) for elem in line.strip().split('\t')]
            user, positive = line[0]
            negatives = line[1:]
            users.extend([user] * 100)
            items.extend([positive] + negatives)
            ratings.extend([1] + [0] * 99)
    return np.array(users), np.array(items), np.array(ratings)


class MovieLensDataModule(pl.LightningDataModule):
    def __init__(
            self,
            variant: str = '1m',
            batch_size: int = 128,
            validate: bool = True,
            n_negatives: int = 8,
            dataset_path: str = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.n_negatives = n_negatives

        self.train_pdf = None
        self.val = None
        self.test = None

        if variant in ['100k', '1m', '10m', '20m']:
            self.validate = validate
            self._download_movielens(variant)
            self.processed = False

        elif variant == 'ncf_dataset':
            if dataset_path is None:
                raise ValueError('Path to dataset must be provided if using ncf_dataset')
            self._read_ncf_dataset(dataset_path)
            self.processed = True

        elif variant == 'from_saved':
            if dataset_path is None:
                raise ValueError('Path to dataset must be provided if using from_saved')
            self._read_saved(dataset_path)
            self.processed = True

        else:
            raise ValueError('Unknown variant')

    def _download_movielens(self, size):
        item_encoder = LabelEncoder()
        user_encoder = LabelEncoder()

        dataset_pdf = movielens.load_pandas_df(
            size=size,
            header=["userID", "itemID", "rating", "timestamp"]
        )

        genres_pdf = movielens.load_item_df(
            size=size,
            genres_col='category',
        )

        genres_pdf = genres_pdf[genres_pdf['itemID'].isin(dataset_pdf['itemID'].unique())]

        dataset_pdf['itemID'] = item_encoder.fit_transform(dataset_pdf['itemID'])
        dataset_pdf['userID'] = user_encoder.fit_transform(dataset_pdf['userID'])

        genres_pdf['itemID'] = item_encoder.transform(genres_pdf['itemID'])
        genres_pdf['category'] = genres_pdf['category'].str.split('|')
        genres_pdf = genres_pdf.explode('category')

        self.abx_tests_pdf = prepare_abx_tests(genres_pdf, item_column_name='itemID', category_column_name='category')

        self.n_items = len(dataset_pdf['itemID'].unique())
        self.n_users = len(dataset_pdf['userID'].unique())

        self.raw_dataset_pdf = dataset_pdf

    def _read_ncf_dataset(self, dataset_path):
        data_root = Path(dataset_path)
        self.train_pdf = pd.read_csv(
            data_root / 'ml-1m.train.rating',
            names=["userID", "itemID", "rating", "timestamp"],
            sep='\t'
        )

        self.val = None

        users, items, ratings = read_test(data_root / 'ml-1m.test.negative')
        self.test = InteractionsDataset(users, items, ratings)

        self.abx_tests_pdf = None
        self.validate = False

        self.n_users = len(np.unique(np.concatenate((self.train_pdf['userID'], users))))
        self.n_items = len(np.unique(np.concatenate((self.train_pdf['itemID'], items))))

    def _read_saved(self, dataset_path):
        data_root = Path(dataset_path)
        with open(data_root / 'stats.txt', 'r') as out_file:
            self.n_users, self.n_items = literal_eval(out_file.read().strip())
        self.train_pdf = pd.read_parquet(data_root / 'train_pdf.parquet')
        self.test = InteractionsDataset.from_saved(data_root / 'test.npz')

        if (data_root / 'abx_tests_pdf.parquet').is_file():
            self.abx_tests_pdf = pd.read_parquet(data_root / 'abx_tests_pdf.parquet')
        else:
            self.abx_tests_pdf = None

        if (data_root / 'val.npz').is_file():
            self.val = InteractionsDataset.from_saved(data_root / 'val.npz')
            self.validate = True
        else:
            self.validate = False

    def save(self, save_dir):
        if not self.processed:
            self.setup()

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / 'stats.txt', 'w') as out_file:
            out_file.write(str((self.n_users, self.n_items)))
        self.train_pdf.to_parquet(save_dir / 'train_pdf.parquet')
        if self.abx_tests_pdf is not None:
            self.abx_tests_pdf.to_parquet(save_dir / 'abx_tests_pdf.parquet')
        if self.val is not None:
            self.val.save(save_dir / 'val.npz')
        self.test.save(save_dir / 'test.npz')

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.processed:
            print("Splitting dataset")
            train_pdf, test_pdf = leave_one_out(self.raw_dataset_pdf)

            users, items, ratings = self._get_augmented_set(
                test_pdf, 99
            )
            self.test = InteractionsDataset(users, items, ratings)

            if self.validate:
                train_pdf, val_pdf = leave_one_out(train_pdf)
                users, items, ratings = self._get_augmented_set(
                    val_pdf, 99
                )
                self.val = InteractionsDataset(users, items, ratings)

            self.train_pdf = train_pdf
            self.processed = True

    def _generate_negative_samples(self, users: np.ndarray, n_samples: int, seed=None) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        users = users.repeat(n_samples).reshape(-1)
        items = rng.integers(self.n_items, size=(len(users),))
        return users, items

    def _get_augmented_set(
            self, pdf: pd.DataFrame, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        users, items = pdf['userID'].values, pdf['itemID'].values
        neg_users, neg_items = self._generate_negative_samples(users, n_samples)

        ratings = np.concatenate((np.ones_like(users), np.zeros_like(neg_users)))
        users, items = np.concatenate((users, neg_users)), np.concatenate((items, neg_items))

        return users, items, ratings

    def train_dataloader(self) -> DataLoader:
        users, items, ratings = self._get_augmented_set(
            self.train_pdf, self.n_negatives
        )
        train = InteractionsDataset(users, items, ratings)
        return DataLoader(train, batch_size=self.batch_size, num_workers=os.cpu_count(), shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.validate:
            return DataLoader(self.val, batch_size=self.batch_size, num_workers=os.cpu_count(), shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=os.cpu_count(), shuffle=False)
