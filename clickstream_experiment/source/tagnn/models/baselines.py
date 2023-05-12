from typing import Union

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class ItemKNNRecommender:
    def __init__(self, k: int = 50):
        self.k = k
        self.train_matrix = None
        self.items_matrix = None
        self.nearest_items = None
        self.num_items = None

    def fit(
            self,
            train_matrix: Union[np.ndarray, csr_matrix],
            item_embeddings: Union[np.ndarray, csr_matrix] = None,
            n_jobs: Union[None, int] = None
    ):
        self.train_matrix = train_matrix
        self.items_matrix = csr_matrix(self.train_matrix.T) if item_embeddings is None else item_embeddings
        self.num_items = self.items_matrix.shape[0]

        nn = NearestNeighbors(n_neighbors=self.k, n_jobs=n_jobs, metric='cosine')
        nn.fit(self.items_matrix)
        dist, neigh = nn.kneighbors(return_distance=True)

        i, j = [], []
        for idx, items in enumerate(neigh):
            i.extend(self.k*[idx])
            j.extend(items)
        i, j = np.array(i), np.array(j)

        scores = 1 - dist.reshape(-1)

        self.nearest_items = csr_matrix((scores, (i, j)), shape=(self.num_items, self.num_items))

    def compute_scores(self, user: int):
        return self.nearest_items.multiply(self.train_matrix[user]).sum(axis=1).A1

    def predict(self, user: int, item: int):
        return self.nearest_items[item].multiply(self.train_matrix[user])

    def get_top_n(self, user: int, topn: int = 50):
        scores = self.compute_scores(user)
        allowed_items = np.setdiff1d(np.arange(self.num_items), self.train_matrix[user].indices, assume_unique=True)

        return allowed_items[np.argsort(scores[allowed_items])[::-1][:topn]]
