import argparse
import numpy as np
from pathlib import Path
from gensim.models import KeyedVectors
from hmmlearn import hmm
from FlowHMM.model.discretized_HMM import DiscreteHMM


DATA_SET = "train"
PROJECT_PATH = f"{Path(__file__).absolute().parent.parent.parent}"


# TODO: save all prints also as information in files


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
    parser.add_argument(
        "--number-of-nodes",
        type=int,
        default=100,
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
        args.number_of_nodes,
    )


def discretize_data(myHMM, w2v_dim, w2v_epochs):
    vectors = KeyedVectors.load(
        f"{PROJECT_PATH}/clickstream_experiment/data/preprocessed_data/vectors_train_{w2v_dim}_{w2v_epochs}.kv"
    )
    vecs = np.concatenate(
        [
            vectors.get_vector(k).reshape(1, -1)
            for k in list(vectors.key_to_index.keys())
        ]
    )
    discrete_index = myHMM.discretize(vecs)

    with open(
        f"{PROJECT_PATH}/clickstream_experiment/data/preprocessed_data/sequences_{w2v_min_len}.txt",
        "r",
    ) as f:
        Xd = [
            np.array(
                [discrete_index[vectors.key_to_index[word]] for word in line.split(" ")]
            ).reshape(-1, 1)
            for line in f.readlines()
        ]
        indexes = np.random.randint(len(Xd), size=220000)
        Xc = [
            np.concatenate([vectors[word].reshape(1, -1) for word in line.split(" ")])
            for i, line in zip(range(len(Xd)), f.readlines())
            if i in indexes
        ]

    Xd_train = [Xd[i] for i in indexes[200000:]]
    Xd_test = [Xd[i] for i in indexes[:200000]]

    Xc_train = [Xc[i] for i in range(200000)]
    Xc_test = [Xc[i] for i in range(200000, 220000)]

    lengths_train = np.array([x.shape[0] for x in Xd_train])
    lengths_sub_train = np.array([x.shape[0] for x in Xc_train])
    lengths_test = np.array([x.shape[0] for x in Xd_test])

    Xd_train = np.concatenate(Xd_train)
    Xd_test = np.concatenate(Xd_test)
    Xc_train = np.concatenate(Xc_train)
    Xc_test = np.concatenate(Xc_test)
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
        number_of_nodes,
    ) = parse_args()
    standardHMM = hmm.GaussianHMM(n_components=n_components, n_iter=150)
    myHMM = DiscreteHMM(
        n_components=n_components,
        learning_alg="cooc",
        discretization_method=discretization_method,
        no_nodes=number_of_nodes,
    )

    (
        Xd_train,
        Xd_test,
        Xc_train,
        Xc_test,
        lengths_train,
        lengths_sub_train,
        lengths_test,
    ) = discretize_data(myHMM, w2v_dim, w2v_epochs)

    standardHMM.fit(Xc_train, lengths_sub_train)
    DiscreteHMM.fit(Xc_test, lengths_test, Xd_train, lengths_train)

    print(
        f"Loglikelihood from standard implementation on test set: {standardHMM.score(Xc_test, lengths_test)}"
    )
    print(
        f"Loglikelihood from my implementation on test set: {myHMM.score(Xc_test, lengths_test)}"
    )
