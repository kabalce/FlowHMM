import argparse
import pickle as pkl
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from clickstream_experiment.source.clickstream import ClickStream
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence
from FlowHMM.model.discretized_HMM import DiscreteHMM


DATA_SET = 'train'
PROJECT_PATH = f'{Path(__file__).absolute().parent.parent.parent}'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w2v-dim", type=int, default=100,
                        help="length of word2vec embedding")
    parser.add_argument("--w2v-epochs", type=int, default=50,
                        help="number of epochs for word2vec training")
    parser.add_argument("--hmm-nodes", type=int, default=500,
                        help="number of epochs for word2vec training")
    parser.add_argument("--w2v-min-len", type=int, default=10,
                        help="minimal sequence length for word3vec training")
    parser.add_argument("--hmm-min-len", type=int, default=10,
                        help="minimal sequence length for HMM training")
    parser.add_argument("--n-components", type=int, default=5,
                        help="number of hidden states for HMM training")
    args = parser.parse_args()
    return args.w2v_dim, args.w2v_epochs, args.hmm_nodes, args.w2v_min_len, args.hmm_min_len, args.n_components


def discretize_data(myHMM, w2v_dim, w2v_epochs):
    # load  embeddings
    vectors = KeyedVectors.load(f'{PROJECT_PATH}/clickstream_experiment/data/preprocessed_data/vectors_train_{w2v_dim}_{w2v_epochs}.kv')
    vecs = None  # vectors to np.array
    # provide grid
    myHMM.provide_nodes(vecs)
    # return grid
    # dict{product_id: node_id}
    # provide sequences of node indexes
    # return Xd
    pass


def train_model():
    # make a workaround to train a model on previously trained data
    pass


if __name__ == "__main__":
    myHMM = DiscreteHMM()

    pass
