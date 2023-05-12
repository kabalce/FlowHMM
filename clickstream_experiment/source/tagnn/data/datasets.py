import numpy as np
from torch.utils.data import Dataset


class InteractionsDataset(Dataset):
    def __init__(
            self,
            users,
            items,
            ratings,
    ):
        self.users = users
        self.items = items
        self.ratings = ratings
        self.n_users = len(np.unique(self.users))
        self.n_items = len(np.unique(self.items))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

    def save(self, path):
        np.savez(path, users=self.users, items=self.items, ratings=self.ratings)

    @classmethod
    def from_saved(cls, path):
        npzfile = np.load(path)
        return cls(npzfile['users'], npzfile['items'], npzfile['ratings'])
