import datetime

from torch_geometric.nn import HGTConv, GATConv, HypergraphConv
from torch_geometric.data import HeteroData
from utils import *
from selfattention import *


class HgTSR(torch.nn.Module):
    def __init__(self, emb_size, incidence_mat, num_heads, num_layers, batch_size, lr):
        super().__init__()
        self.lr = lr
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.incidence_mat = incidence_mat
        self.n_node = self.incidence_mat.shape[0]
        self.n_edge = self.incidence_mat.shape[1]


        self.nodes_emb = nn.Parameter(torch.Tensor(self.n_node + 1, self.emb_size))
        self.edges_emb = nn.Parameter(torch.Tensor(self.n_edge, self.emb_size))

        self.H_GAT = HypergraphConv(self.emb_size, self.emb_size)

        self.encoders = nn.ModuleList([EncoderLayer(self.emb_size, 100, 10, 10, 10) for _ in range(1)])

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, node2edge_index, reversed_sess_item, mask, batch_size, edge2edge):

        # positional encoding
        # data_form['nodes'].x = self.position_enc(data_form.x_dict['nodes'])

        # data_form['nodes'].x = data_form.x_dict['nodes'][sessions != 0]

        # nodes_x = self.HgT(data_form)
        nodes_x = self.H_GAT(self.nodes_emb, node2edge_index, hyperedge_attr=self.edges_emb)

        get = lambda i: nodes_x[reversed_sess_item[i]]

        seq_h = torch.cuda.FloatTensor(batch_size, reversed_sess_item.shape[1], self.emb_size).fill_(0)
        for i in torch.arange(reversed_sess_item.shape[0]):
            seq_h[i] = get(i)
        mask = mask.float().unsqueeze(-1)


        for enc_layer in self.encoders:
            seq_h = enc_layer(seq_h, slf_attn_mask=mask)[0]

        select = seq_h[:, :1, :]
        select = select.contiguous().view(-1, self.emb_size)

        return select, nodes_x


class HyperGraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, data_form, num_heads, num_layers):
        super().__init__()


        self.conv = nn.ModuleList([Node2Edge(in_channels, out_channels, data_form, num_heads),
                                   Edge2Node(in_channels, out_channels, data_form, num_heads)

                                   ])


    def forward(self, data_form):
        for i, c in enumerate(self.conv):
            # x_dict = c(x_dict, edge_index_dict)
            # x_nodes, x_edges = data_form.x_dict['nodes'], data_form.x_dict['edges']
            if i % 2 == 0:
                data_form['edges'].x = c(data_form.x_dict, data_form.edge_index_dict)
            else:
                data_form['nodes'].x = c(data_form.x_dict, data_form.edge_index_dict)

        return data_form.x_dict['nodes']


class Node2Edge(torch.nn.Module):
    def __init__(self, in_channels, out_channels, data_form, num_heads):
        super().__init__()
        self.conv = HGTConv(in_channels, out_channels, data_form.metadata(), num_heads, group='sum')

    def forward(self, x_dict, edge_index_dict):
        temp_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            _, rel, _ = edge_type
            if rel == 'aggregate':
                temp_dict['nodes', 'aggregate', 'edges'] = edge_index

        x_dict = self.conv(x_dict, temp_dict)

        return x_dict['edges']


class Edge2Node(torch.nn.Module):
    def __init__(self, in_channels, out_channels, data_form, num_heads):
        super().__init__()
        self.conv = HGTConv(in_channels, out_channels, data_form.metadata(), num_heads, group='sum')

    def forward(self, x_dict, edge_index_dict):
        temp_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            _, rel, _ = edge_type
            if rel == 'send':
                temp_dict['edges', 'send', 'nodes'] = edge_index

        x_dict = self.conv(x_dict, temp_dict)
        return x_dict['nodes']


class LineConv(torch.nn.Module):
    def __init__(self, layers, batch_size, emb_size=100):
        super(LineConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers
        self.gat = nn.ModuleList([GATConv(self.emb_size, self.emb_size) for _ in range(2)])

    def forward(self, item_embedding, edge_index, session_item, session_len):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        item_embedding = torch.cat([zeros, item_embedding[1:]], 0)
        seq_h = []
        for i in torch.arange(len(session_item)):
            seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
        session_emb_lgcn = torch.div(torch.sum(seq_h1, 1), session_len)
        session = [session_emb_lgcn]
        if len(edge_index) == 0:
            return session_emb_lgcn
        for gat in self.gat:
            session_emb_lgcn = gat(session_emb_lgcn, edge_index)
            session.append(session_emb_lgcn)
        session_emb_lgcn = torch.div(np.sum(session, 0), 3)
        return session_emb_lgcn


def train_processing(model, i, data, batch_size, node2edge_index, edge2edge):
    tar, session_len, sessions, reversed_sess_item, mask = data.get_slice(i)
    tar = trans_to_cuda(torch.LongTensor(tar))
    mask = trans_to_cuda(torch.LongTensor(mask))
    reversed_sess_item = trans_to_cuda(torch.LongTensor(reversed_sess_item))

    sessions_emb, items_emb = model(node2edge_index, reversed_sess_item, mask, batch_size, edge2edge)
    scores = torch.mm(sessions_emb, torch.transpose(items_emb[1:], 1, 0))
    return tar, scores


def test_processing(model, i, data, batch_size, node2edge_index, edge2edge):
    tar, session_len, sessions, reversed_sess_item, mask = data.get_slice(i)
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.LongTensor(reversed_sess_item))

    sessions_emb, items_emb = model(node2edge_index, reversed_sess_item, mask, batch_size, edge2edge)
    scores = torch.mm(sessions_emb, torch.transpose(items_emb[1:], 1, 0))

    return tar, scores


def train_test(model, train_data, test_data, batch_size):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0

    model.train()
    slices = train_data.generate_batch(batch_size)

    node2edge_index = trans_to_cuda(torch.LongTensor(train_data.get_all_node2edge()).t().contiguous())


    print('getting the edge2edge...')
    # edge2edge = get_overlap(train_data.raw)
    edge2edge = None

    for i in slices:
        model.zero_grad()
        tar, scores = train_processing(model, i, train_data, batch_size, node2edge_index, edge2edge)
        loss = model.loss_function(scores + 1e-8, tar)
        print(loss)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()

    print('\tLoss:\t%.3f' % total_loss)
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    slices = test_data.generate_batch(batch_size)
    for i in slices:
        tar, scores = test_processing(model, i, test_data, batch_size, node2edge_index, edge2edge)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
    return metrics, total_loss
