import argparse
import pickle
import time
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/Nowplaying/sample')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')

opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))
    all_train_seq = pickle.load(open('./datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))

    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 40727
    elif opt.dataset == 'Nowplaying':
        n_node = 60416
    else:
        n_node = 309

    train_data = Data(train_data, all_train_seq, shuffle=True)
    test_data = Data(test_data, all_train_seq, shuffle=True)
    model = trans_to_cuda(GHCSR(hg_adjacency=train_data.hg_edge, gnn_adjacency=train_data.gnn_edge,
                                emb_size=opt.embSize, n_node=n_node,
                                num_heads=4, num_layers=opt.layer, batch_size=opt.batchSize, lr=opt.lr))

    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    f = open('./result.txt', 'w')
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data, opt.batchSize)
        print('current epoch: ', epoch)
        print('Recall@20:\t%.4f\tMMR@20:\t%.4f' % (hit, mrr))
        print('current epoch: ', epoch, file=f)
        print('Recall@20:\t%.4f\tMMR@20:\t%.4f', hit, mrr, file=f)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))
    f.close()


if __name__ == '__main__':
    main()
