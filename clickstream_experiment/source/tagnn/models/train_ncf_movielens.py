from time import time

import click
import numpy as np
import pandas as pd
import wandb
import yaml
from recommenders.models.ncf.dataset import Dataset as NCFDataset

from src.data.movielens import get_movielens, get_movielens_abx
from src.models.ncf import NCF


@click.command()
@click.argument('config_filepath', type=click.Path(exists=True))
@click.option('--repeats', default=1)
def main(config_filepath, repeats):
    with open(config_filepath, 'r') as config_file:
        cfg = yaml.load(config_file, yaml.loader.Loader)

    if cfg['seed'] == -1:
        cfg['seed'] = np.random.randint(100000)

    name = (
            cfg.get('model_type', 'NeuMF')
            + f"_ml{cfg['movielens_size']}"
            + ("_mlp" if cfg.get('init_mlp_embeddings', False) else "")
            + (
                    ("_gmf" if cfg.get('init_gmf_embeddings', False) else "")
                    + ("_combined" if cfg.get('combined_gmf', False) else "")
                    + ("_frozen" if cfg.get('freeze_gmf', False) else "")
                    + (f"_{cfg['unfreeze_gmf']}" if cfg.get('unfreeze_gmf', None) is not None else "")
            )
            + (f"_late_gmf_{cfg['late_init_gmf']}" if cfg.get('late_init_gmf', None) is not None else "")
    )

    validate = cfg.get('validate', True)

    if cfg['movielens_size'] != 'ncf_paper':
        train_pdf, val_pdf, test_pdf = get_movielens(cfg['movielens_size'], validate=validate)
        if validate:
            data = NCFDataset(train=train_pdf, test=val_pdf, seed=cfg['data_seed'])
        else:
            data = NCFDataset(train=train_pdf, test=test_pdf, seed=cfg['data_seed'])
    else:
        train_pdf = pd.read_csv('data/processed/ml-1m.train.rating')
        test_pdf = pd.read_csv('data/processed/ml-1m.test.rating')
        data = NCFDataset(train=train_pdf, test=test_pdf, seed=cfg['data_seed'])

    abx_pdf = get_movielens_abx(cfg['movielens_size'], seed=cfg['data_seed'], items_to_keep=data.item2id.keys(),
                                item2id=data.item2id)

    for _ in range(repeats):
        wandb.init(project="ncf", entity="cirglaboratory", name=name, config=cfg, reinit=True)

        model = train_model(cfg, data, abx_pdf)

        model.save(f"/Ziob/adurb/recommender_system/models/ncf/{name}")

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
        combined_gmf=cfg.get('combined_gmf', False)
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
