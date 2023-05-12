import wandb

from src.data.movielens import get_movielens, get_movielens_abx
from src.models.embeddings.item2vec import Item2Vec

cfg = {
    'movielens_size': '100k',
    'vector_size': 8,
    'window': 15,
    'alpha': 0.035,
    'epochs': 20,
}

wandb.init(config=cfg)

cfg = wandb.config


def main():
    train_pdf, test_pdf = get_movielens(cfg['movielens_size'])

    abx_pdf = get_movielens_abx(cfg['movielens_size'], seed=42, items_to_keep=train_pdf['itemID'].unique())

    model = Item2Vec(
        vector_size=cfg['vector_size'],
        window=cfg['window'],
        alpha=cfg['alpha'],
        epochs=cfg['epochs'],
    )
    model.train(train_pdf, user_col='userID', item_col='itemID')

    wandb.log(
        model.calculate_abx_scores(abx_pdf)
    )


if __name__ == '__main__':
    main()
