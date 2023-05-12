import pickle
from collections import defaultdict
from itertools import permutations

import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.data.abx import prepare_abx_tests
from src.models.embeddings.item2vec import Item2Vec
from src.utils.utils import normalize, flatten


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion, seed=None):
    rng = np.random.default_rng(seed)
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    rng.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def _get_local_graph(max_n_node, u_input, node):
    u_A = np.zeros((max_n_node, max_n_node))
    for i in np.arange(len(u_input) - 1):
        if u_input[i + 1] == 0:
            break
        u = np.where(node == u_input[i])[0][0]
        v = np.where(node == u_input[i + 1])[0][0]
        u_A[u][v] = 1
    return _get_graph_in_out(u_A)


def _get_global_graph(max_n_node, node, edge_weights):
    u_A = np.zeros((max_n_node, max_n_node))
    for u, v in permutations(node, 2):
        u_idx = np.where(node == u)[0][0]
        v_idx = np.where(node == v)[0][0]
        u_A[u_idx][v_idx] = edge_weights[(u, v)]
    return _get_graph_in_out(u_A)

def _get_graph_in_out(u_A):
    u_sum_in = np.sum(u_A, 0)
    u_sum_in[np.where(u_sum_in == 0)] = 1
    u_A_in = np.divide(u_A, u_sum_in)
    u_sum_out = np.sum(u_A, 1)
    u_sum_out[np.where(u_sum_out == 0)] = 1
    u_A_out = np.divide(u_A.transpose(), u_sum_out)
    u_A = np.concatenate([u_A_in, u_A_out]).transpose()
    return u_A


class Data:
    def __init__(self, data, shuffle=False, edge_weights=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.edge_weights = edge_weights

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input[u_input != 0])))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input[u_input != 0])
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            if self.edge_weights is None:
                u_A = _get_local_graph(max_n_node, u_input, node)
            else:
                u_A = _get_global_graph(max_n_node, node, self.edge_weights)
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] if i != 0 else 0 for i in u_input])
        return alias_inputs, A, items, mask, targets


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def find_edge_weights(sequences):
    edge_weights = defaultdict(lambda: 0)
    for seq in sequences:
        seq_edges = set()
        for edge in zip(seq[:-1], seq[1:]):
            seq_edges.add(edge)
        for edge in seq_edges:
            edge_weights[edge] += 1
    return edge_weights


def read_data(data_path, dataset, validation, valid_portion):
    train_data = pickle.load(open(data_path / f"{dataset}/TAGNN_seq_5.pkl", 'rb'))
    if validation:
        train_data, valid_data = split_validation(train_data, valid_portion, seed=42)
        test_data = valid_data
    else:
        test_data = pickle.load(open(data_path / f"{dataset}/TAGNN_seq_5_TEST.pkl", 'rb'))

    clicks_pdf = pd.read_csv(
        data_path / 'TAGNN_df_5_train.dat',
        header=None,
        names=['sessionID', 'timestamp', 'itemID', 'category'],
        parse_dates=['timestamp'],
        dtype={'sessionID': int, 'itemID': int, 'category': str},
    )

    # item2id = pickle.load(open(data_path / "item2id.txt", 'rb'))
    # id2item = {new_id: item_id for item_id, new_id in item2id.items()}
    train_data = pickle.load(open(data_path / "train.txt", 'rb'))

    dates = pickle.load(open(data_path / "yoochoose1_64/dates.txt", 'rb'))
    dates = pd.to_datetime(dates, unit='s')
    if dates.tz is None:
        dates = dates.tz_localize('UTC')
    max_date = max(dates)

    items_in_train = np.unique(list(flatten(train_data))).astype('int64')

    filtered_clicks_pdf = clicks_pdf[clicks_pdf['timestamp'] < max_date]

    long_sessions = filtered_clicks_pdf['sessionID'].value_counts() >= 2
    long_sessions = long_sessions.index[long_sessions]

    filtered_clicks_pdf = filtered_clicks_pdf[filtered_clicks_pdf['sessionID'].isin(long_sessions)]

    item2id = {i: i for i in items_in_train}

    return train_data, test_data, filtered_clicks_pdf, items_in_train, item2id


def calculate_embeddings(opt, clicks_pdf, items_in_train, item2id, n_node):
    embedding_model = Item2Vec(vector_size=opt.hiddenSize)
    embedding_model.train(clicks_pdf, item_col='itemID', user_col='sessionID', epochs=3)
    embeddings_pdf = embedding_model.generate_item_embeddings()
    embeddings_pdf = embeddings_pdf.loc[embeddings_pdf.index.isin(items_in_train)]
    embeddings_pdf = pd.DataFrame(normalize(embeddings_pdf.values), index=embeddings_pdf.index)

    embeddings_pdf.index = map(lambda item_id: item2id[str(item_id)], embeddings_pdf.index)

    embeddings = np.random.standard_normal((n_node, opt.hiddenSize))
    embeddings[embeddings_pdf.index] = embeddings_pdf.values

    return embeddings


def calculate_abx(clicks_pdf, items_in_train, item2id):
    categories_pdf = clicks_pdf[clicks_pdf['itemID'].isin(items_in_train)][['itemID', 'category']].drop_duplicates()
    categories_pdf = categories_pdf[categories_pdf['category'].isin(map(str, range(1, 13)))]
    categories_pdf.loc[:, 'itemID'] = list(map(lambda item_id: item2id[str(item_id)], categories_pdf['itemID']))
    abx_tests_pdf = prepare_abx_tests(categories_pdf, item_column_name='itemID', category_column_name='category', seed=42)
    return abx_tests_pdf