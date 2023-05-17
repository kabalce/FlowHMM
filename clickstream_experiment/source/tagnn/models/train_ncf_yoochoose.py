from pathlib import Path
from time import time

import click
import numpy as np
import pandas as pd
import wandb
import yaml
from recommenders.models.ncf.dataset import Dataset as NCFDataset

from src.data.abx import prepare_abx_tests
from src.data.split import leave_one_out
from src.data.utils import filter_dataset
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
            + f"_yoochoose"
            + ("_mlp" if cfg.get('init_mlp_embeddings', False) else "")
            + (
                    ("_gmf" if cfg.get('init_gmf_embeddings', False) else "")
                    + ("_frozen" if cfg.get('freeze_gmf', False) else "")
                    + (f"_{cfg['unfreeze_gmf']}" if cfg.get('unfreeze_gmf', None) is not None else "")
            )
            + (f"_late_gmf_{cfg['late_init_gmf']}" if cfg.get('late_init_gmf', None) is not None else "")
    )

    data_path = Path(cfg['dataset_path'])

    clicks_pdf = pd.read_csv(
        data_path / 'yoochoose-clicks.dat',
        header=None,
        names=['sessionID', 'timestamp', 'itemID', 'category'],
        parse_dates=['timestamp'],
        dtype={'sessionID': int, 'itemID': int, 'category': str},
        )

    random_items = np.random.choice(clicks_pdf['itemID'].unique(), 25000, replace=False)

    clicks_pdf = clicks_pdf[clicks_pdf['itemID'].isin(random_items)]
    clicks_pdf = clicks_pdf.iloc[-int(len(clicks_pdf) / 64):]

    filtered_clicks_pdf = filter_dataset(clicks_pdf, user_col='sessionID')
    filtered_clicks_pdf['rating'] = 1
    train_pdf, test_pdf = leave_one_out(filtered_clicks_pdf, user_col='sessionID')

    data = NCFDataset(train=train_pdf, test=test_pdf, seed=cfg['data_seed'], col_user='sessionID')
    if cfg.get('precomputed_abx', False):
        print('Using precomputed abx')
        abx_pdf = pd.read_parquet(data_path / 'abx_tests.parquet')
    else:
        categories_pdf = clicks_pdf[
            (clicks_pdf['itemID'].isin(data.item2id.keys()))
            & (clicks_pdf['category'].isin(map(str, range(1, 13))))
            ][['itemID', 'category']].drop_duplicates()

        categories_pdf.loc[:, 'itemID'] = list(map(lambda item: data.item2id[item], categories_pdf.loc[:, 'itemID']))

        abx_pdf = prepare_abx_tests(categories_pdf, category_column_name='category', item_column_name='itemID')

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
    )

    model.init_data(data)

    log_data = model.evaluate(data, abx_tests_pdf=abx_tests_pdf, user_col='sessionID')

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
            log_data = model.evaluate(data, abx_tests_pdf=abx_tests_pdf, user_col='sessionID')
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
