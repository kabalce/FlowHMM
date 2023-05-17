from typing import Sequence

import numpy as np
import torch
from torch import nn, optim

from src.models.lightning_model import BaseModel
from src.models.matrix_factorization.utils import generate_embeddings


class MLP(nn.Module):
    def __init__(
            self,
            n_users: int,
            n_items: int,
            layer_sizes: Sequence[int],
    ):
        assert layer_sizes[0] % 2 == 0

        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.hidden = layer_sizes[0] // 2

        self.user_encoder = nn.Embedding(n_users, layer_sizes[0]//2)
        self.item_encoder = nn.Embedding(n_items, layer_sizes[0]//2)

        linear_layers = []
        layer_sizes = list(layer_sizes)
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            linear_layers.append(nn.Linear(in_size, out_size))

        self.linear_layers = nn.ModuleList(linear_layers)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        assert len(users) == len(items)
        output = torch.concat((self.user_encoder(users), self.item_encoder(items)), dim=-1)
        for layer in self.linear_layers:
            output = torch.relu(layer(output))
        return output

    def init_embeddings(self, train_pdf, device):
        item_embeddings = generate_embeddings(
            train_pdf, self.n_items, self.hidden, user_col='userID', item_col='itemID'
        )
        self.item_encoder = nn.Embedding.from_pretrained(torch.Tensor(item_embeddings), freeze=True).to(device)

    def unfreeze_embeddings(self):
        self.item_encoder.requires_grad_(True)

    def get_item_embeddings(self) -> np.ndarray:
        return self.item_encoder.weight.cpu().detach().numpy()

    def get_embeddings_parameters(self):
        return list(self.item_encoder.parameters())


class MultiLayerPerceptron(BaseModel):
    def __init__(
            self,
            n_users: int,
            n_items: int,
            layer_sizes: Sequence[int],
            lr: float = 1e-3,
            lr_decay: float = 0.1,
            lr_step_size: int = 16,
            weight_decay: float = 0,
            init_embeddings: bool = False,
            unfreeze_after: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_users = n_users
        self.n_items = n_items

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_step_size = lr_step_size
        self.weight_decay = weight_decay

        self.init_embeddings = init_embeddings
        self.unfreeze_after = unfreeze_after
        self.abx_tests_pdf = None

        self.mlp = MLP(n_users, n_items, layer_sizes)
        self.linear = nn.Linear(layer_sizes[-1], 1)

        for weight in self.parameters():
            nn.init.normal_(weight, 0, 0.1)

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        self.hr_best: float = 0
        self.ndcg_best: float = 0

    def configure_optimizers(self):
        # parameters = (
        #         list(self.user_encoder.parameters()) + list(self.linear_layers.parameters())
        # )
        # if not self.init_embeddings:
        #     parameters.extend(list(self.item_encoder.parameters()))
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        output = self.mlp(users, items)
        output = self.linear(output)
        return output.reshape(-1)

    def _get_item_embeddings(self) -> np.ndarray:
        return self.mlp.get_item_embeddings()

    def on_train_start(self) -> None:
        self.abx_tests_pdf = self.datamodule.abx_tests_pdf
        if self.init_embeddings:
            self.mlp.init_embeddings(self.datamodule.train_pdf, self.device)

    def on_train_epoch_start(self) -> None:
        if self.init_embeddings and self.current_epoch == self.unfreeze_after:
            self.mlp.unfreeze_embeddings()
            self.optimizers().add_param_group({
                'params': self.mlp.get_embeddings_parameters()
            })
