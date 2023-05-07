import argparse
import numpy as np
from pathlib import Path
from gensim.models import KeyedVectors
from hmmlearn import hmm
import logging
import sys
from icecream import ic
import pickle as pkl
import datetime


DATA_SET = "train"
PROJECT_PATH = f"{Path(__file__).absolute().parent.parent.parent}"
sys.path.insert(1, PROJECT_PATH)

from torchHMM.model.discretized_HMM import DiscreteHMM

np.random.seed(2023)
# TODO: add progress logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--w2v-dim", type=int, default=100, help="length of word2vec embedding"
    )
    parser.add_argument(
        "--w2v-epochs",
        type=int,
        default=50,
        help="number of epochs for word2vec training",
    )
    parser.add_argument(
        "--hmm-nodes",
        type=int,
        default=500,
        help="number of epochs for word2vec training",
    )
    parser.add_argument(
        "--w2v-min-len",
        type=int,
        default=10,
        help="minimal sequence length for word3vec training",
    )
    parser.add_argument(
        "--hmm-min-len",
        type=int,
        default=10,
        help="minimal sequence length for HMM training",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=5,
        help="number of hidden states for HMM training",
    )
    parser.add_argument(
        "--discrete-meth",
        type=str,
        default="uniform",
        help="discretization technique for hmm training",
    )
    args = parser.parse_args()
    return (
        args.w2v_dim,
        args.w2v_epochs,
        args.hmm_nodes,
        args.w2v_min_len,
        args.hmm_min_len,
        args.n_components,
        args.discrete_meth,
    )


def discretize_data(myHMM, w2v_dim, w2v_epochs, w2v_min_len):
    data_path = f"{PROJECT_PATH}/clickstream_experiment/data/preprocessed_data/train_valid_data_{w2v_dim}_{w2v_epochs}_{w2v_min_len}.pkl"
    if Path(data_path).exists():
        with open(data_path, 'rb') as f:
            data = pkl.load(f)
        myHMM.nodes = data['myHMM.nodes']
        return (
            data['Xd_train'],
            data['Xd_test'],
            data['Xc_train'],
            data['Xc_test'],
            data['lengths_train'],
            data['lengths_sub_train'],
            data['lengths_test'],
        )
    else:
        vectors = KeyedVectors.load(
            f"{PROJECT_PATH}/clickstream_experiment/data/preprocessed_data/vectors_train_{w2v_dim}_{w2v_min_len}_{w2v_epochs}.kv"
        )
        vecs = np.concatenate(
            [
                vectors.get_vector(k).reshape(1, -1)
                for k in list(vectors.key_to_index.keys())
            ]
        )

        myHMM.provide_nodes(vecs, force=False)
        batch_size = 25000
        discrete_index = np.concatenate(
            [
                myHMM.discretize(
                    vecs[(batch_size * i) : (batch_size * (i + 1))], force=False
                )
                for i in range(vecs.shape[0] // batch_size + 1)
            ]
        )

        print("discretized.")

        with open(
            f"{PROJECT_PATH}/clickstream_experiment/data/preprocessed_data/sequences_{w2v_min_len}.txt",
            "r",
        ) as f:
            Xd = [
                np.array(
                    [
                        discrete_index[vectors.key_to_index[word]]
                        for word in line.replace("\n", "").split(" ")
                    ]
                ).reshape(-1, 1)
                for line in f.readlines()
            ]
            subsample_size = 100000
            indexes = np.random.choice(
                len(Xd), size=int(subsample_size * 1.1), replace=False
            )
            f.seek(0)
            indexes_l = indexes.tolist()
            Xc = [
                np.concatenate(
                    [
                        vectors[word].reshape(1, -1)
                        for word in line.replace("\n", "").split(" ")
                    ]
                )
                for i, line in enumerate(f)
                if i in indexes_l
            ]

        Xd_train = [Xd[i] for i in range(len(Xd)) if i not in indexes[subsample_size:]]
        Xd_test = [Xd[i] for i in indexes[subsample_size:]]

        Xc_train = [Xc[i] for i in indexes.argsort()[:subsample_size]]
        Xc_test = [Xc[i] for i in indexes.argsort()[subsample_size:]]

        lengths_train = np.array([x.shape[0] for x in Xd_train])
        lengths_sub_train = np.array([x.shape[0] for x in Xc_train])
        lengths_test = np.array([x.shape[0] for x in Xc_test])

        Xd_train = np.concatenate(Xd_train)
        Xd_test = np.concatenate(Xd_test)
        Xc_train = np.concatenate(Xc_train)
        Xc_test = np.concatenate(Xc_test)

        results = {
            'Xd_train': Xd_train,
            'Xd_test': Xd_test,
            'Xc_train': Xc_train,
            'Xc_test': Xc_test,
            'lengths_train': lengths_train,
            'lengths_sub_train': lengths_sub_train,
            'lengths_test': lengths_test,
            'myHMM.nodes': myHMM.nodes
        }

        with open(data_path, 'wb') as f:
            pkl.dump(results, f)

        return (
            Xd_train,
            Xd_test,
            Xc_train,
            Xc_test,
            lengths_train,
            lengths_sub_train,
            lengths_test,
        )


if __name__ == "__main__":
    (
        w2v_dim,
        w2v_epochs,
        hmm_nodes,
        w2v_min_len,
        hmm_min_len,
        n_components,
        discretization_method,
    ) = parse_args()

    logging.basicConfig(
        filename=f"{PROJECT_PATH}/clickstream_experiment/logs/train_hmm_{w2v_dim}_{w2v_epochs}_{w2v_min_len}_{hmm_nodes}_{n_components}_{discretization_method}.log",
        encoding="utf-8",
        level=logging.DEBUG,
    )

    standardHMM = hmm.GaussianHMM(n_components=n_components, n_iter=150)
    myHMM = DiscreteHMM(
        n_components=n_components,
        learning_alg="cooc",
        discretization_method=discretization_method,
        no_nodes=hmm_nodes,
    )

    (
        Xd_train,
        Xd_test,
        Xc_train,
        Xc_test,
        lengths_train,
        lengths_sub_train,
        lengths_test,
    ) = discretize_data(myHMM, w2v_dim, w2v_epochs, w2v_min_len)

    myHMM.fit(X=Xc_test, lengths=lengths_test, Xd=Xd_train, lengths_d=lengths_train)

    print(
        f"Mean loglikelihood from my implementation on test set: {myHMM.score(Xc_test, lengths_test) / Xc_test.shape[0]}"
    )
    logging.debug(
        f"Mean loglikelihood from my implementation on test set: {myHMM.score(Xc_test, lengths_test) / Xc_test.shape[0]}"
    )

    standardHMM.fit(Xc_train, lengths_sub_train)

    print(
        f"Mean loglikelihood from standard implementation on test set: {standardHMM.score(Xc_test, lengths_test) / Xc_test.shape[0]}"
    )
    logging.debug(
        f"Mean loglikelihood from standard implementation on test set: {standardHMM.score(Xc_test, lengths_test) / Xc_test.shape[0]}"
    )
