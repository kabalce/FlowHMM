from abc import abstractmethod

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from src.data.abx import calculate_abx_score
from src.utils.evaluation import hit_rate_from_ranks, ndcg_from_ranks


def find_ranks(users, ratings, predictions):
    ratings_pdf = pd.DataFrame({
        'userID': users,
        'rating': ratings,
        'prediction': predictions,
    })

    ratings_pdf = ratings_pdf.sort_values('rating', ascending=True)
    ratings_pdf = ratings_pdf.sort_values(['userID', 'prediction'], ascending=False)
    ratings_pdf['rank'] = np.arange(len(ratings_pdf))

    lowest_rank = ratings_pdf.drop_duplicates(['userID'], keep='first').set_index('userID')['rank']
    true_rank = ratings_pdf[ratings_pdf['rating'] == 1].set_index('userID')['rank']

    ranks = true_rank.sort_index().values - lowest_rank.sort_index().values

    return ranks


def _get_metrics_from_outputs(outputs):
    users = np.concatenate([res['users'] for res in outputs])
    ratings = np.concatenate([res['ratings'] for res in outputs])
    predictions = np.concatenate([res['predictions'] for res in outputs])

    ranks = find_ranks(users, ratings, predictions)

    hr: float = hit_rate_from_ranks(ranks)
    ndcg: float = ndcg_from_ranks(ranks)

    return hr, ndcg


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.hr_best: float = 0
        self.ndcg_best: float = 0

    @abstractmethod
    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_item_embeddings(self) -> np.ndarray:
        pass

    def training_step(self, batch, batch_idx):
        users, items, ratings = batch

        output = self(users, items)
        loss = self.criterion(output.float(), ratings.float())

        self.log("loss", loss)

        return loss

    def _predict_batch(self, batch):
        users, items, ratings = batch
        predictions = self(users, items)

        results = {
            "users": users.cpu(),
            "ratings": ratings.cpu(),
            "predictions": predictions.cpu(),
        }

        return results

    def on_train_epoch_end(self) -> None:
        if self.abx_tests_pdf is not None:
            abx_score = calculate_abx_score(self._get_item_embeddings(), self.abx_tests_pdf)
            self.log_dict(abx_score)

    def test_step(self, batch, batch_index):
        return self._predict_batch(batch)

    def validation_step(self, batch, batch_index):
        return self._predict_batch(batch)

    def validation_epoch_end(self, outputs) -> None:
        hr, ndcg = _get_metrics_from_outputs(outputs)

        self.hr_best = max(self.hr_best, hr)
        self.ndcg_best = max(self.ndcg_best, ndcg)

        self.log_dict({
            'hr_val': hr,
            'ndcg_val': ndcg,
            'hr_best': self.hr_best,
            'ndcg_best': self.ndcg_best,
        })

    def test_epoch_end(self, outputs) -> None:
        hr, ndcg = _get_metrics_from_outputs(outputs)

        self.log_dict({
            'hr_test': hr,
            'ndcg_test': ndcg,
        })
