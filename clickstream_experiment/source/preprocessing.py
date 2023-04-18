import argparse
import pickle as pkl
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
import sys

PROJECT_PATH = f"{Path(__file__).absolute().parent.parent.parent}"
sys.path.insert(1, PROJECT_PATH)
from clickstream_experiment.source.clickstream import ClickStream


DATA_SET = "train"
logging.basicConfig(
    filename=f"{PROJECT_PATH}/clickstream_experiment/logs/preprocessing.log",
    encoding="utf-8",
    level=logging.DEBUG,
)

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
        "--w2v-min-len",
        type=int,
        default=10,
        help="minimal sequence length for word3vec training",
    )
    args = parser.parse_args()
    return args.w2v_dim, args.w2v_epochs, args.w2v_min_len


def load_raw_clickstream():
    with open(
        f"{PROJECT_PATH}/clickstream_experiment/data/raw_data/{DATA_SET}.jsonl"
    ) as f:
        click_stream = ClickStream(f)
    with open(
        f"{PROJECT_PATH}/clickstream_experiment/data/preprocessed_data/ClickStream_{DATA_SET}.pkl",
        "wb",
    ) as f:
        pkl.dump(click_stream, f)
    return click_stream


def analyze_clickstream(click_stream):
    # TODO: analiza typów sesji

    session_lens = pd.Series(click_stream.session_lengths)
    item_ids = pd.Series(click_stream.item_ids)

    print(f"Number of sessions in {DATA_SET} set: {len(click_stream.sessions)}")
    logging.debug(f"Number of sessions in {DATA_SET} set: {len(click_stream.sessions)}")
    print(f"Number of products in {DATA_SET} set: {item_ids.shape[0]}")
    logging.debug(f"Number of products in {DATA_SET} set: {item_ids.shape[0]}")
    print(
        f"Average session length in {DATA_SET} set: {(session_lens.index * session_lens).sum() / session_lens.sum()}"
    )  # TODO: fix the statistics
    logging.debug(f"Average session length in {DATA_SET} set: {session_lens.mean()}")

    session_lens.sort_index().plot(kind="bar")
    plt.savefig(
        f"{PROJECT_PATH}/clickstream_experiment/analysis/sesstion_lens_{DATA_SET}.png"
    )

    item_ids.value_counts().sort_index().plot(kind="hist")
    plt.xlabel("Number of occurrences")
    plt.ylabel("Number of products")
    plt.savefig(
        f"{PROJECT_PATH}/clickstream_experiment/analysis/item_freqs_{DATA_SET}.png"
    )


def prepare_file_for_w2v(click_stream, w2v_min_len):
    with open(
        f"{PROJECT_PATH}/clickstream_experiment/data/preprocessed_data/sequences_{w2v_min_len}.txt",
        "w",
    ) as f:
        [
            f.write(" ".join([str(w) for w in s.list_items()]) + "\n")
            for s in click_stream.sessions
            if len(s.event_list) >= w2v_min_len
        ]


def train_w2v(w2v_dim, w2v_epochs, w2v_min_len):
    w2v_model = Word2Vec(
        sentences=LineSentence(
            f"{PROJECT_PATH}/clickstream_experiment/data/preprocessed_data/sequences_{w2v_min_len}.txt"
        ),
        vector_size=w2v_dim,
        window=5,
        min_count=1,
        workers=30,
        epochs=w2v_epochs,
    )
    w2v_model.wv.save(
        f"{PROJECT_PATH}/clickstream_experiment/data/preprocessed_data/vectors_train_{w2v_dim}_{w2v_min_len}_{w2v_epochs}.kv"
    )


if __name__ == "__main__":
    # logging.debug('Start preprocessing.')
    w2v_dim, w2v_epochs, w2v_min_len = parse_args()

    cs_path = f"{PROJECT_PATH}/clickstream_experiment/data/preprocessed_data/ClickStream_{DATA_SET}.pkl"
    w2v_path = f"{PROJECT_PATH}/clickstream_experiment/data/preprocessed_data/sequences_{w2v_min_len}.txt"
    if not os.path.exists(w2v_path):
        if os.path.exists(cs_path):
            with open(
                cs_path,
                "rb",
            ) as f:
                cs = pkl.load(f)
        else:
            cs = load_raw_clickstream()

        # analyze_clickstream(cs)

        """
        Number of sessions in train set: 12899779
        Number of products in train set: 1855603
        Average session length in train set: 16.799985178040647
        """

        logging.info("ClickStream loaded.")
        prepare_file_for_w2v(cs, w2v_min_len)
        logging.info("Data for word2vec written to file.")
        del cs

    train_w2v(w2v_dim, w2v_epochs, w2v_min_len)
