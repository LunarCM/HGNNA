import datetime
import math
import torch.nn.functional as F
from utils import *
from torch import nn
from torch.nn import Module
from torch_geometric.nn.conv import GCNConv, HypergraphConv


class HyperConv(Module):
    def __init__(self, layers, emb_size=100):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.hg_conv = nn.ModuleList([HypergraphConv(self.emb_size, self.emb_size) for _ in range(self.layers)])

    def forward(self, adjacency, embedding):
        item_embeddings = embedding
        final = [item_embeddings.unsqueeze(0)]
        for conv in self.hg_conv:
            item_embeddings = conv(item_embeddings, trans_to_cuda(adjacency))
            final.append(item_embeddings.unsqueeze(0))
        final = torch.cat(final, dim=0)
        item_embeddings = torch.sum(final, dim=0) / (self.layers + 1)
        return item_embeddings


class ItemConv(Module):
    def __init__(self, layers, emb_size=100):
        super(ItemConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.gcn = nn.ModuleList([GCNConv(self.emb_size, self.emb_size) for _ in range(self.layers - 1)])

    def forward(self, adjacency, embedding):
        item_embeddings = embedding
        final = [item_embeddings.unsqueeze(0)]
        for conv in self.gcn:
            item_embeddings = conv(item_embeddings, trans_to_cuda(adjacency))
            final.append(item_embeddings.unsqueeze(0))
        final = torch.cat(final, dim=0)
        item_embeddings = torch.sum(final, dim=0) / self.layers
        return item_embeddings


class GHCSR(torch.nn.Module):
    def __init__(self, hg_adjacency, gnn_adjacency, emb_size, n_node, num_heads, num_layers, batch_size, lr):
        super().__init__()
        self.hg_adjacency = hg_adjacency
        self.gnn_adjacency = gnn_adjacency
        self.lr = lr
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.n_node = n_node
        self.nodes_emb = nn.Embedding(self.n_node, self.emb_size)

        self.HyperGnn = HyperConv(self.num_layers)
        self.Gnn = ItemConv(self.num_layers)
        self.w = nn.Linear(2 * self.emb_size, self.emb_size)

        self.pos_embedding = nn.Embedding(150, self.emb_size)
        self.w_1 = nn.Linear(2 * self.emb_size, self.emb_size)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.lstm = nn.LSTM(self.emb_size, hidden_size=2 * self.emb_size, num_layers=num_layers - 1, batch_first=True)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def soft_attention(self, seq_h, mask, session_len):
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)

        return select

    def contrastive_learning(self, features):
        labels = torch.cat([torch.arange(features.shape[0] / 2) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = trans_to_cuda(labels)

        similarity_matrix = torch.matmul(features, features.T)
        flag = trans_to_cuda(torch.eye(labels.shape[0], dtype=torch.bool))
        labels = labels[~flag].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~flag].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = trans_to_cuda(torch.zeros(logits.shape[0], dtype=torch.long))
        logits = logits / 0.2
        con_loss = self.loss_function(logits, labels)
        return 0.001 * con_loss

    def forward(self, tar, session_item, reversed_sess_item, mask, session_len, is_train):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        item_embeddings = torch.cat([zeros, self.nodes_emb.weight], 0)

        item_embeddings_hg = self.HyperGnn(self.hg_adjacency, item_embeddings)
        # item_embeddings_hg = F.normalize(item_embeddings_hg, dim=-1, p=2)
        item_embeddings_gnn = self.Gnn(self.gnn_adjacency, item_embeddings)
        # item_embeddings_gnn = F.normalize(item_embeddings_gnn, dim=-1, p=2)
        # item_embeddings = self.w(torch.cat([item_embeddings_hg, item_embeddings_gnn], -1))
        # item_embeddings = torch.tanh(item_embeddings)
        # item_embeddings = (item_embeddings_hg + item_embeddings_gnn) / 2

        get_hg = lambda i: item_embeddings_hg[reversed_sess_item[i]]
        seq_h_hg = torch.cuda.FloatTensor(self.batch_size, reversed_sess_item.shape[1], self.emb_size).fill_(0)
        for i in torch.arange(reversed_sess_item.shape[0]):
            seq_h_hg[i] = get_hg(i)
        get_gnn = lambda i: item_embeddings_gnn[reversed_sess_item[i]]
        seq_h_gnn = torch.cuda.FloatTensor(self.batch_size, reversed_sess_item.shape[1], self.emb_size).fill_(0)
        for i in torch.arange(reversed_sess_item.shape[0]):
            seq_h_gnn[i] = get_gnn(i)

        # get_item = lambda i: item_embeddings[session_item[i]]
        # seq_h = torch.cuda.FloatTensor(self.batch_size, reversed_sess_item.shape[1], self.emb_size).fill_(0)
        # for i in torch.arange(reversed_sess_item.shape[0]):
        #     seq_h[i] = get_item(i)

        select_hg = self.soft_attention(seq_h_hg, mask, session_len)
        select_gnn = self.soft_attention(seq_h_gnn, mask, session_len)

        if is_train:
            # hg_items = seq_h_hg.contiguous().view(-1, self.emb_size)
            # gnn_items = seq_h_gnn.contiguous().view(-1, self.emb_size)
            # features = torch.cat([hg_items, gnn_items], dim=0)
            # con_loss = self.contrastive_learning(features)
            # print(con_loss)
            con_loss = 0
        else:
            con_loss = 0

        # seq_h = (seq_h_hg + seq_h_gnn) / 2
        # seq_h = seq_h_hg

        # output, _ = self.lstm(seq_h)




        scores = torch.mm(select, torch.transpose(item_embeddings[1:], 1, 0))
        loss = self.loss_function(scores + 1e-8, tar)

        return con_loss, loss, scores


def processing(model, i, data, is_train):
    tar, session_len, session_item, reversed_sess_item, mask = data.get_slice(i)
    tar = trans_to_cuda(torch.LongTensor(tar))
    session_item = trans_to_cuda(torch.LongTensor(session_item))
    reversed_sess_item = trans_to_cuda(torch.LongTensor(reversed_sess_item))
    session_len = trans_to_cuda(torch.LongTensor(session_len))
    mask = trans_to_cuda(torch.LongTensor(mask))
    if is_train:
        con_loss, tar_loss, scores = model(tar, session_item, reversed_sess_item, mask, session_len, is_train=True)
        return con_loss, tar_loss
    else:
        con_loss, tar_loss, scores = model(tar, session_item, reversed_sess_item, mask, session_len, is_train=False)
        return tar, scores


def train_test(model, train_data, test_data, batch_size):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    model.train()
    slices = train_data.generate_batch(batch_size)
    for b, i in enumerate(slices):
        model.zero_grad()
        con_loss, tar_loss, = processing(model, i, train_data, is_train=True)
        # loss = con_loss + tar_loss
        loss = tar_loss
        print(b, con_loss, tar_loss)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(batch_size)
    for i in slices:
        tar, scores = processing(model, i, test_data, is_train=False)
        tar = trans_to_cpu(tar).detach().numpy()
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target in zip(sub_scores, tar):
            hit.append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
