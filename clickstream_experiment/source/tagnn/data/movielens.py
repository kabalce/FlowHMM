from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_chrono_split

from src.data.abx import prepare_abx_tests
from src.data.split import leave_one_out


def get_movielens(movielens_size, validate=True):
    ml_pdf = movielens.load_pandas_df(
        size=movielens_size,
        header=["userID", "itemID", "rating", "timestamp"]
    )

    # PROCESS DATASET
    train_pdf, test_pdf = leave_one_out(ml_pdf)
    if validate:
        train_pdf, val_pdf = leave_one_out(ml_pdf)
    else:
        val_pdf = None

    return train_pdf, val_pdf, test_pdf


def get_movielens_abx(movielens_size, seed, items_to_keep=None, item2id=None):
    items_df = movielens.load_item_df(movielens_size, genres_col='genre')
    items_df['genre'] = items_df.apply(lambda row: row['genre'].split('|'), axis=1)
    items_df = items_df.explode('genre')
    if items_to_keep is not None:
        items_df = items_df[items_df['itemID'].isin(items_to_keep)]
    if item2id is not None:
        items_df.loc[:, 'itemID'] = list(map(lambda item: item2id[item], items_df.loc[:, 'itemID']))

    return prepare_abx_tests(items_df, category_column_name='genre', item_column_name='itemID', seed=seed)
