import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from gensim import matutils
from gensim.models import Word2Vec
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope


def recall(top_n, test_items):
    return len(set(top_n) & set(test_items)) / len(test_items)


def get_top_n(self, word2_indices, allowed_indices, topn=10):

    l1 = np.sum(self.wv.vectors[word2_indices], axis=0)
    if word2_indices and self.cbow_mean:
        l1 /= len(word2_indices)

    # propagate hidden -> output and take softmax to get probabilities
    prob_values = np.exp(np.dot(l1, self.syn1neg[allowed_indices].T))
    prob_values /= np.sum(prob_values)
    top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)

    # returning the most probable output words
    return [self.wv.index_to_key[allowed_indices[index1]] for index1 in top_indices]


Word2Vec.get_top_n = get_top_n


def test_model(model, users, type, n=50):
    all_items = set(map(model.wv.get_index, train_pdf['asin'].unique()))

    metric_vals = []

    for user in users:
        user_train_items = set(map(model.wv.get_index, train_items_pdf.loc[user]))
        allowed_items = all_items - user_train_items

        if type == 'items':
            top_n = model.get_top_n(list(user_train_items), list(allowed_items), topn=n)
        elif type == 'user':
            top_n = model.get_top_n([model.wv.get_index(user)], list(allowed_items), topn=n)
        else:
            raise ValueError("unknown type")

        user_test_items = test_items_pdf.loc[user]

        metric_vals.append(recall(top_n, user_test_items))

    return np.mean(metric_vals)


def hyperopt_train_test(params):
    model = Word2Vec(
        corpus_file=str(word2vec_data_path / 'recommender_item_item_permutations.cor'),
        workers=4,
        **params
    )
    return test_model(model, test_users[:1000], 'items')


def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}


if __name__ == '__main__':
    project_path = Path('/pio/scratch/1/i308362/recommender_system')
    data_path = project_path / 'data' / 'processed' / 'amazon-books' / '5-core'
    word2vec_data_path = project_path / 'data' / 'processed' / 'word2vec' / 'amazon-books' / '5-core'

    train_pdf = pd.read_parquet(data_path / 'train.parquet')
    test_pdf = pd.read_parquet(data_path / 'test.parquet')

    test_users = test_pdf['reviewerID'].unique()

    train_items_pdf = train_pdf[train_pdf['reviewerID'].isin(test_users)].groupby('reviewerID').apply(lambda pdf: pdf['asin'].unique())
    test_items_pdf = test_pdf.groupby('reviewerID').apply(lambda pdf: pdf['asin'].unique())

    space = {
        'vector_size': scope.int(hp.quniform('vector_size', 50, 250, 10)),
        'window': scope.int(hp.quniform('window', 5, 20, 1)),
        'alpha': hp.uniform('aplha', 0.01, 0.04),
        'epochs': scope.int(hp.quniform('epochs', 3, 20, 1)),
    }

    trials = Trials()
    fmin(f, space, algo=tpe.suggest, max_evals=50, trials=trials)

    with open(project_path / 'models' / 'word2vec' / 'hyperopt_trials.pickle', 'wb') as f_out:
        pickle.dump(trials, f_out)
