import argparse
import pickle
import numpy as np
from pathlib import Path
import sys
import wandb

PROJECT_PATH = "/ziob/klaudia/FlowHMM"
sys.path.insert(1, PROJECT_PATH)

from clickstream_experiment.source.tagnn.model import SessionGraph
from clickstream_experiment.source.tagnn.wrappers import Trainer
from clickstream_experiment.source.tagnn.utils import (
    Data,
    trans_to_cuda,
    split_validation,
    normalize,
    flatten,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/ziob/klaudia/FlowHMM/clickstream_experiment/data/preprocessed_data",
    )
    parser.add_argument("--batchSize", type=int, default=32, help="input batch size")
    parser.add_argument("--hiddenSize", type=int, default=100, help="hidden state size")
    parser.add_argument(
        "--epoch", type=int, default=30, help="the number of epochs to train for"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate"
    )  # [0.001, 0.0005, 0.0001]
    parser.add_argument(
        "--lr_dc", type=float, default=0.1, help="learning rate decay rate"
    )
    parser.add_argument(
        "--lr_dc_step",
        type=int,
        default=3,
        help="the number of steps after which the learning rate decay",
    )
    parser.add_argument(
        "--l2", type=float, default=1e-5, help="l2 penalty"
    )  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument("--step", type=int, default=1, help="gnn propogation steps")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="the number of epoch to wait before early stop ",
    )
    parser.add_argument("--variant", default="hybrid")
    parser.add_argument("--ignore_target", action="store_true")
    parser.add_argument("--validation", action="store_true", help="validation")
    parser.add_argument(
        "--valid_portion",
        type=float,
        default=0.1,
        help="split the portion of training set as validation set",
    )
    parser.add_argument("--device_ids", type=int, nargs="*")
    parser.add_argument(
        "--init_embeddings",
        action="store_true",
        help="initialize embeddings using word2vec",
    )
    parser.add_argument(
        "--unfreeze_embeddings",
        type=int,
        default=1,
        help="epoch in which to unfreeze the embeddings layer",
    )
    parser.add_argument("--name", type=str, help="name of wandb run")
    parser.add_argument("--use_global_graph", action="store_true")
    parser.add_argument(
        "--log_freq", type=int, default=5, help="how many times to log during an epoch"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="/ziob/klaudia/FlowHMM/clickstream_experiment/logs/",
        help="file to save model results in",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/ziob/klaudia/FlowHMM/clickstream_experiment/data/models",
        help="directory to save model weights in",
    )
    parser.add_argument(
        "--dim", type=int, default=100, help="length of word2vec embedding"
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=10,
        help="minimal sequence length for word3vec training",
    )
    opt = parser.parse_args()
    # Path(opt.model_dir).mkdir(exist_ok=True)
    # Path(opt.results_path).mkdir(exist_ok=True)
    return opt


def read_data(opt):
    data_path = Path(opt.data_path)

    data_path, validation, valid_portion = data_path, opt.validation, opt.valid_portion
    train_data = pickle.load(
        open(f"{data_path}/TAGNN_seq_{opt.min_len}_train.pkl", "rb")
    )
    if validation:
        train_data, valid_data = split_validation(train_data, valid_portion, seed=42)
        test_data = valid_data
    else:
        test_data = pickle.load(
            open(f"{data_path}/TAGNN_seq_{opt.min_len}_test.pkl", "rb")
        )

    items_in_train = np.unique(flatten(train_data) + flatten(test_data)).astype("int64")

    item2id = {items_in_train[i]: i for i in range(len(items_in_train))}
    id2item = {i: items_in_train[i] for i in range(len(items_in_train))}

    n_node = items_in_train.shape[0]

    # TODO: obsłuż nierozpoznane indeksy
    train_data = [[item2id[i] for i in s] for s in train_data[0]], [
        item2id[i] for i in train_data[1]
    ]
    test_data = [[item2id[i] for i in s] for s in test_data[0]], [
        item2id[i] for i in test_data[1]
    ]
    return train_data, test_data, item2id, id2item, n_node


def train_model(opt, train_data, test_data, n_node):
    model = SessionGraph(opt, n_node, init_embeddings=None)
    model = trans_to_cuda(model).float()

    trainer = Trainer(model, log_freq=opt.log_freq, model_dir=opt.model_dir)
    trainer.fit(Data(train_data), Data(test_data), opt.epoch, opt.patience)
    return trainer


# TODO: Zapisz embeddingi i cały model


def save_results(trainer, data, results_path, id2item):
    trainer.save_results(data, results_path, id2item)


if __name__ == "__main__":
    opt = parse_args()
    train_data, test_data, item2id, id2item, n_node = read_data(opt)
    trainer = train_model(opt, train_data, test_data, n_node)
    save_results(trainer, train_data, opt.results_path, id2item)
