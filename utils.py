import torch
import numpy as np


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


# return each session has what item
def data_masks_1(all_sessions):
    edge_index = []
    for i, session in enumerate(all_sessions):
        if len(session) < 2:
            continue
        for item in session:
            edge_index.append([item, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return edge_index


def data_masks_2(all_sessions):
    temp = set()
    for session in all_sessions:
        for i in range(len(session) - 1):
            temp.add((session[i], session[i + 1]))
            temp.add((session[i] + 1, session[i]))
    edge_index = []
    for t in temp:
        edge_index.append([t[0], t[1]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return edge_index


class Data:
    def __init__(self, data, all_train_seq, shuffle=False):
        self.raw = np.asarray(data[0])
        self.shuffle = shuffle
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.hg_edge = data_masks_1(self.raw)
        self.gnn_edge = data_masks_2(all_train_seq)

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
        # print(max_n_node)
        # if max_n_node > 64:
        #     max_n_node = 64
        #     for i, n in enumerate(num_node):
        #         if n > 64:
        #             inp[i] = inp[i][n - 64:-1]
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
