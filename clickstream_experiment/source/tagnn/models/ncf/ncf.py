from typing import Sequence, Tuple, Optional

import numpy as np
import torch
from torch import nn, optim

from src.data.abx import calculate_abx_score
from src.models.lightning_model import BaseModel
from src.models.ncf.gmf import GMF
from src.models.ncf.mlp import MLP


class NeuralCollaborativeFiltering(BaseModel):
    def __init__(
            self,
            n_users: int,
            n_items: int,
            hidden: int,
            layer_sizes: Sequence[int],
            lr: float = 1e-3,
            lr_decay: float = 0.1,
            lr_step_size: int = 16,
            weight_decay: float = 0,
            init_embeddings: bool = False,
            unfreeze_after: int = 10,
            pretrained_path: Optional[str] = None,
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

        self.mlp = MLP(n_users, n_items, layer_sizes)
        self.gmf = GMF(n_users, n_items, hidden)
        if pretrained_path is not None:
            self.mlp.load_state_dict(torch.load(pretrained_path + '/mlp.pt'))
            self.gmf.load_state_dict(torch.load(pretrained_path + '/gmf.pt'))
        self.linear = nn.Linear(hidden+layer_sizes[-1], 1)

        for weight in self.parameters():
            nn.init.normal_(weight, 0, 0.1)

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        self.hr_best: float = 0
        self.ndcg_best: float = 0

    def configure_optimizers(self):
        parameters = []
        for name, param in self.named_parameters():
            if not name.startswith('gmf.item_encoder'):
                parameters.append(param)
        if not self.init_embeddings:
            parameters.extend(self.gmf.get_embeddings_parameters())

        optimizer = optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf_out = self.gmf(users, items)
        mlp_out = self.mlp(users, items)
        output = torch.concat((gmf_out, mlp_out), dim=-1)
        output = self.linear(output)
        return output.reshape(-1)

    def _get_item_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mlp.get_item_embeddings(), self.gmf.get_item_embeddings()

    def on_train_epoch_end(self) -> None:
        if self.abx_tests_pdf is not None:
            mlp_embeddings, gmf_embeddings = self._get_item_embeddings()
            mlp_score = calculate_abx_score(mlp_embeddings, self.abx_tests_pdf)
            gmf_score = calculate_abx_score(gmf_embeddings, self.abx_tests_pdf)
            self.log_dict({
                'abx_euclidean_mlp': mlp_score['abx_euclidean'],
                'abx_cosine_mlp': mlp_score['abx_cosine'],
                'abx_euclidean_gmf': gmf_score['abx_euclidean'],
                'abx_cosine_gmf': gmf_score['abx_cosine'],
            })

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
