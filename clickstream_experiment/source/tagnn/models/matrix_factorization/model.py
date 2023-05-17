import numpy as np
import torch
from torch import nn, optim

from src.models.lightning_model import BaseModel
from src.models.matrix_factorization.utils import generate_embeddings


class MatrixFactorization(BaseModel):
    def __init__(
            self,
            n_users: int,
            n_items: int,
            hidden: int,
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
        self.hidden = hidden

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_step_size = lr_step_size
        self.weight_decay = weight_decay

        self.init_embeddings = init_embeddings
        self.unfreeze_after = unfreeze_after
        self.abx_tests_pdf = None

        self.user_encoder = nn.Embedding(n_users, hidden)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_encoder = nn.Embedding(n_items, hidden)
        self.item_bias = nn.Embedding(n_items, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        for weight in self.parameters():
            nn.init.normal_(weight, 0, 0.1)

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def configure_optimizers(self):
        parameters = (
                [self.bias] + list(self.user_encoder.parameters())
                + list(self.user_bias.parameters()) + list(self.item_bias.parameters())
        )
        if not self.init_embeddings:
            parameters.extend(list(self.item_encoder.parameters()))
        optimizer = optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def predict_all(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        dot_product = self.user_encoder(users)@self.item_encoder(items).T
        bias = self.user_bias(users) + self.item_bias(items).T + self.bias
        return dot_product + bias

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        assert len(users) == len(items)
        dot_product = torch.sum(self.user_encoder(users)*self.item_encoder(items), dim=1)
        bias = self.user_bias(users).reshape(-1) + self.item_bias(items).reshape(-1) + self.bias
        return dot_product + bias

    def _get_item_embeddings(self) -> np.ndarray:
        return self.item_encoder.weight.cpu().detach().numpy()

    def on_train_start(self) -> None:
        self.abx_tests_pdf = self.datamodule.abx_tests_pdf
        if self.init_embeddings:
            item_embeddings = generate_embeddings(
                self.datamodule.train_pdf, self.datamodule.n_items, self.hidden, user_col='userID', item_col='itemID'
            )
            self.item_encoder = nn.Embedding.from_pretrained(torch.Tensor(item_embeddings), freeze=True).to(self.device)

    def on_train_epoch_start(self) -> None:
        if self.init_embeddings and self.current_epoch == self.unfreeze_after:
            self.item_encoder.requires_grad_(True)
            self.optimizers().add_param_group({
                'params': list(self.item_encoder.parameters())
            })
