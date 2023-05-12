import argparse
import pickle
from pathlib import Path

import wandb
import sys

PROJECT_PATH = f"{Path(__file__).absolute().parent.parent.parent.parent}"
sys.path.insert(1, PROJECT_PATH)
from clickstream_experiment.source.tagnn.model import SessionGraph
from clickstream_experiment.source.tagnn.wrappers import Trainer
from clickstream_experiment.source.tagnn.utils_ import (
    Data, trans_to_cuda, find_edge_weights, read_data, calculate_embeddings, calculate_abx
)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/Ziob/klaudia/FlowHMM/clickstream_experiment/data/')
parser.add_argument('--dataset', default='processed_data', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')
parser.add_argument('--variant', default='hybrid')
parser.add_argument('--ignore_target', action='store_true')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--device_ids', type=int, nargs='*')
parser.add_argument('--init_embeddings', action='store_true', help='initialize embeddings using word2vec')
parser.add_argument('--unfreeze_embeddings', type=int, default=1, help='epoch in which to unfreeze the embeddings layer')
parser.add_argument('--name', type=str, help='name of wandb run')
parser.add_argument('--use_global_graph', action='store_true')
parser.add_argument('--log_freq', type=int, default=5, help='how many times to log during an epoch')
parser.add_argument('--results_path', type=str, help='file to save model results in')
parser.add_argument('--model_dir', type=str, help='directory to save model weights in')
opt = parser.parse_args()
print(opt)


def main():
    if opt.variant not in ['hybrid', 'local', 'global', 'none']:
        raise ValueError('Unknown variant')

    data_path = Path(opt.data_path)

    dataset_version = data_path.parts[-1]
    name = f"TAGNN_{opt.dataset}_{dataset_version}_train" if opt.name is None else opt.name
    wandb.init(project="tagnn", entity="cirglaboratory", name=name, config=opt)

    if opt.model_dir is not None:
        save_path = Path(opt.model_dir) / 'opt.pickle'
        with open(save_path, 'wb') as out_file:
            pickle.dump(opt, out_file)
        wandb.save(str(save_path))

    train_data, test_data, filtered_clicks_pdf, items_in_train, item2id = \
        read_data(data_path, opt.dataset, opt.validation, opt.valid_portion)

    n_node = len(item2id) + 1

    if opt.init_embeddings:
        embeddings = calculate_embeddings(opt, filtered_clicks_pdf, items_in_train, item2id, n_node)
        print("Embeddings calculated")
    else:
        embeddings = None

    abx_tests_pdf = calculate_abx(filtered_clicks_pdf, items_in_train, item2id)

    if opt.use_global_graph:
        all_train_seq = pickle.load(open(data_path / f"{opt.dataset}/all_train_seq.txt", 'rb'))
        edge_weights = find_edge_weights(all_train_seq)
        train_data = Data(train_data, shuffle=True, edge_weights=edge_weights)
    else:
        train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)

    model = SessionGraph(opt, n_node, init_embeddings=embeddings)
    model = trans_to_cuda(model).float()

    trainer = Trainer(model, log_freq=opt.log_freq, model_dir=opt.model_dir)
    unfreeze_embeddings = opt.unfreeze_embeddings if opt.init_embeddings else None
    trainer.fit(train_data, test_data, opt.epoch, opt.patience, unfreeze_embeddings, abx_tests_pdf)

    wandb.finish()


if __name__ == '__main__':
    main()
