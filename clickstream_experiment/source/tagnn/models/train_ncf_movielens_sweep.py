from time import time

import numpy as np
import wandb
from recommenders.models.ncf.dataset import Dataset as NCFDataset

from src.data.movielens import get_movielens, get_movielens_abx
from src.models.ncf import NCF

cfg = {
    'movielens_size': '100k',
    'epochs': 200,
    'top_k': 10,
    'batch_size': 256,
    'seed': np.random.randint(100000),
    'data_seed': 42,
    'learning_rate': 1e-4,
    'layer_sizes': [32, 16, 8, 4],
    'n_factors': 8,
    'log_freq': 10,
}

wandb.init(config=cfg)

cfg = wandb.config


def main():
    train_pdf, test_pdf = get_movielens(cfg['movielens_size'])

    data = NCFDataset(train=train_pdf, test=test_pdf, seed=cfg['data_seed'])

    abx_pdf = get_movielens_abx(cfg['movielens_size'], seed=cfg['data_seed'], items_to_keep=data.item2id.keys(), item2id=data.item2id)

    train_model(cfg, data, abx_pdf)

    wandb.finish()


def train_model(cfg, data, abx_tests_pdf):

    model = NCF(
        n_users=data.n_users,
        n_items=data.n_items,
        seed=cfg['seed'],
        embeddings_data=data,
        model_type=cfg.get('model_type', 'NeuMF'),
        n_factors=cfg['n_factors'],
        layer_sizes=cfg['layer_sizes'],
        n_epochs=cfg['epochs'],
        batch_size=cfg['batch_size'],
        learning_rate=float(cfg['learning_rate']),
        verbose=cfg.get('log_freq', 10),
        init_mlp_embeddings=cfg.get('init_mlp_embeddings', False),
        init_gmf_embeddings=cfg.get('init_gmf_embeddings', False),
        freeze_gmf=cfg.get('freeze_gmf', False),
    )

    model.init_data(data)

    log_data = model.evaluate(data, abx_tests_pdf=abx_tests_pdf)

    wandb.log(
        log_data
    )

    unfreeze_gmf = cfg.get('unfreeze_gmf', None)

    # loop for n_epochs
    for epoch_count in range(1, model.n_epochs + 1):

        train_begin = time()

        if unfreeze_gmf is None or epoch_count < unfreeze_gmf:
            train_loss = model.run_epoch(data, unfreeze_gmf=False)
        else:
            train_loss = model.run_epoch(data, unfreeze_gmf=True)

        train_time = time() - train_begin

        # output every self.verbose
        if model.verbose and (epoch_count % model.verbose == 0 or epoch_count == model.n_epochs - 1):
            log_data = model.evaluate(data, abx_tests_pdf=abx_tests_pdf)
        else:
            log_data = {}

        log_data['train_loss'] = sum(train_loss) / len(train_loss)
        log_data['train_time'] = train_time

        wandb.log(
            log_data
        )

    return model


if __name__ == '__main__':
    main()
