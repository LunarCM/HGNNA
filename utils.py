import math

import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix


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


class RelTemporalEncoding(nn.Module):

    def __init__(self, n_hid, max_len=240, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t))


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


# return each session has what item
def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        # drop duplication
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i] - 1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))

    return matrix


class Data:
    def __init__(self, data, shuffle=False, n_node=None):
        self.raw = np.asarray(data[0])
        self.incidence_mat = data_masks(self.raw, n_node).T
        self.shuffle = shuffle
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)

    def get_all_node2edge(self):
        node2edge = []
        for i, session in enumerate(self.raw):
            for item in set(session):
                node2edge.append([item, i])
        return node2edge

    def get_all_edge2node(self):
        edge2node = []
        for i, session in enumerate(self.raw):
            for item in set(session):
                edge2node.append([i, item])
        return edge2node

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            # pad 0 at behind
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            # [1, 1, ... ,1 , 0 , 0, ...]
            mask.append([1] * len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
        return self.targets[index] - 1, session_len, items, reversed_sess_item, mask


# return [num_edges, 2]
def get_node2edge(session_index, sessions):
    node2edge_index = []
    for i, session in enumerate(sessions):
        for item in session:
            if item != 0:
                node2edge_index.append([item, session_index[i]])
    return node2edge_index


def get_edge2node(session_index, sessions):
    node2edge_index = []
    for i, session in enumerate(sessions):
        for item in session:
            if item != 0:
                node2edge_index.append([item, session_index[i]])
    return node2edge_index


def get_node2edge_(sessions):
    node2edge_index = []
    j = 0
    for i, session in enumerate(sessions):
        for item in session:
            if item != 0:
                node2edge_index.append([j, i])
                j += 1
    return node2edge_index


def get_edge2node_(sessions):
    edge2node_index = []
    j = 0
    for i, session in enumerate(sessions):
        for item in session:
            if item != 0:
                edge2node_index.append([i, j])
                j += 1
    return edge2node_index


def get_overlap(sessions):
    edeg_index = []
    for i in range(len(sessions)):
        seq_a = set(sessions[i])
        seq_a.discard(0)
        for j in range(i + 1, len(sessions)):
            seq_b = set(sessions[j])
            seq_b.discard(0)
            overlap = seq_a.intersection(seq_b)
            if len(overlap) > 0:
                edeg_index.append([i, j])
                edeg_index.append([j, i])
    return edeg_index
