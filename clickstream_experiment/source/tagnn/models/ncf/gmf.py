import numpy as np
import torch
from torch import nn, optim

from src.models.lightning_model import BaseModel
from src.models.matrix_factorization.utils import generate_embeddings


class GMF(nn.Module):
    def __init__(
            self,
            n_users: int,
            n_items: int,
            hidden: int,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.hidden = hidden

        self.user_encoder = nn.Embedding(n_users, hidden)
        self.item_encoder = nn.Embedding(n_items, hidden)


    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        assert len(users) == len(items)
        elementwise_product = self.user_encoder(users)*self.item_encoder(items)
        return elementwise_product

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


class GeneralizedMatrixFactorization(BaseModel):
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

        self.gmf = GMF(n_users, n_items, hidden)
        self.linear = nn.Linear(hidden, 1)

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_step_size = lr_step_size
        self.weight_decay = weight_decay
        self.init_embeddings = init_embeddings

        self.unfreeze_after = unfreeze_after
        self.abx_tests_pdf = None

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        for weight in self.parameters():
            nn.init.normal_(weight, 0, 0.1)

    def configure_optimizers(self):
        parameters = (
                list(self.gmf.user_encoder.parameters()) + list(self.linear.parameters())
        )
        if not self.init_embeddings:
            parameters.extend(list(self.gmf.item_encoder.parameters()))
        optimizer = optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        output = self.gmf(users, items)
        output = self.linear(output)
        return output.reshape(-1)

    def _get_item_embeddings(self) -> np.ndarray:
        return self.gmf.get_item_embeddings()

    def on_train_start(self) -> None:
        self.abx_tests_pdf = self.datamodule.abx_tests_pdf
        if self.init_embeddings:
            self.gmf.init_embeddings(self.datamodule.train_pdf, self.device)

    def on_train_epoch_start(self) -> None:
        if self.init_embeddings and self.current_epoch == self.unfreeze_after:
            self.gmf.unfreeze_embeddings()
            self.optimizers().add_param_group({
                'params': self.gmf.get_embeddings_parameters()
            })
