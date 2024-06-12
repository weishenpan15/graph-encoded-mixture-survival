import matplotlib.pyplot as plt
from clusterGNN import ClusterGNN, GNNModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import h5py
from sklearn.cluster import KMeans
import argparse
import os.path as osp
import torch.nn as nn
from sksurv.metrics import concordance_index_censored
from utils import cluster_logrank, eval_logrank
import json
from get_graph import *

def train_test_split_from_file(f, censor_flag, ratio, seed):
    train_x, train_t, train_e = np.array(f['train']['x']), np.array(f['train']['t']), np.array(f['train']['e'])
    test_x, test_t, test_e = np.array(f['test']['x']), np.array(f['test']['t']), np.array(f['test']['e'])

    if censor_flag == "uncensor":
        train_x, train_t, train_e = train_x[train_e > 0], train_t[train_e > 0], train_e[train_e > 0]
        test_x, test_t, test_e = test_x[test_e > 0], test_t[test_e > 0], test_e[test_e > 0]

    all_x, all_t, all_e = np.concatenate((train_x, test_x)), np.concatenate((train_t, test_t)), np.concatenate(
        (train_e, test_e))
    train_x, test_x, train_e, test_e, train_t, test_t = train_test_split(all_x, all_e, all_t, test_size=1 - ratio,
                                                                         shuffle=True, random_state=seed)
    all_x, all_e, all_t = np.concatenate((train_x, test_x)), np.concatenate((train_e, test_e)), np.concatenate(
        (train_t, test_t))

    index = list(range(all_x.shape[0]))
    # np.random.shuffle(index)
    index1, index2 = index[:int(ratio * all_x.shape[0])], index[int(ratio * all_x.shape[0]):]
    train_num = len(index1)
    test_num = len(index2)

    return all_x, all_t, all_e, index1, index2, train_num, test_num


if __name__ == '__main__':
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=2, type=int, help='num of cluster')
    parser.add_argument('--seed', default=0, type=int, help='random_seed')
    parser.add_argument('--dataset', default='metabric', type=str, help='dataset')
    parser.add_argument('--censor_flag', default='censor', type=str, help='censor if including censor data')
    parser.add_argument('--lr', default=0.005, type=float, help='')
    parser.add_argument('--kcore', default=10, type=int, help='')
    parser.add_argument('--pre_emb', default=0, type=int, help='')
    parser.add_argument('--pre_k', default=1, type=int, help='')
    parser.add_argument('--modify_nsc', default=1, type=int, help='')
    parser.add_argument('--inductive', default=1, type=int, help='')

    # no emb 0: has k 1, modify nsc 1;
    # no emb 0: no k 0, ori nsc 0 working worse
    # has emb 1: has k 1, modify nsc 1
    # has emb 1: no k 0, ori nsc 0 working worse

    args = parser.parse_args()

    seed = args.seed
    dataset = args.dataset
    censor_flag = args.censor_flag

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load and split data
    if dataset == "metabric":
        f = h5py.File('./data/metabric_IHC4_clinical_train_test.h5', 'r')
    else:
        raise NotImplementedError

    all_x, all_t, all_e, index1, index2, train_num, test_num \
        = train_test_split_from_file(f, censor_flag, 0.8, args.seed)

    train_x, train_t, train_e = all_x[index1], all_t[index1], all_e[index1]
    test_x, test_t, test_e = all_x[index2], all_t[index2], all_e[index2]

    # normalization
    stder = StandardScaler()
    stder.fit(train_x)
    all_x_standard = stder.transform(all_x)
    train_x_standard = all_x_standard[index1]
    test_x_standard = all_x_standard[index2]

    minmax = lambda x: x / train_t.max()  # Enforce to be inferior to 1
    train_t_ddh = minmax(train_t)

    para_dir = "saved_clustergnn_para_ind"
    with open(osp.join(para_dir, "{}_{}_seed_{}".format(dataset, censor_flag, seed)), "r") as fin:
        best_para_dict = json.loads(fin.read())
    print("Best Parameter Set:", best_para_dict)

    if not args.inductive:
        graph,  _ = get_graph(best_para_dict['kcore'], all_x_standard, all_x_standard, train_t_ddh, train_e)
    else:
        graph, adj = get_graph(best_para_dict['kcore'], train_x_standard, train_x_standard, train_t_ddh, train_e)

    cgnn = ClusterGNN(epochs=best_para_dict['epochs'], num_workers=0, lr=args.lr, graph=graph, random_state=seed, para_dict=best_para_dict)

    cgnn.fit(graph, index1)

    # testing
    if not args.inductive:
        test_graph = graph
        train_survival, train_alphas = cgnn.predict_survival(graph, index1)
        test_survival, test_alphas = cgnn.predict_survival(test_graph, index2)
    else:
        test_graph = get_test_graph(best_para_dict['kcore'], train_x_standard, graph, test_x_standard)
        train_survival, train_alphas = cgnn.predict_survival(graph, index1)
        test_survival, test_alphas = cgnn.predict_survival(test_graph, index2)

    # C index
    train_c_idx = concordance_index_censored(train_e.astype(bool), train_t, 1 - train_survival[:, 0])
    test_c_idx = concordance_index_censored(test_e.astype(bool), test_t, 1 - test_survival[:, 0])
    print("Prediction C Index: Train {}, Test {}".format(train_c_idx[0], test_c_idx[0]))

    # the evaluation with the significance for multivariate_logrank_test
    train_c = np.argmax(train_alphas.detach().cpu().numpy()[:, 0, :], axis=1)
    test_c = np.argmax(test_alphas.detach().cpu().numpy()[:, 0, :], axis=1)

    print("Clustering Train:")
    train_c, trainstat_list, trainlogp_list = eval_logrank(train_c, train_t, train_e)
    for c in range(args.k):
        print("\tTrain Cluster {}: Mean Survival Time {}, Censor Rate {}, Number {}".format(c, train_t[
            train_c == c].mean(), train_e[train_c == c].mean(), (train_c == c).sum()))
    print(
        "\tAverage Train: tstats {}, logp {}".format(np.array(trainstat_list).mean(), np.array(trainlogp_list).mean()))

    print("Clustering Test:")
    test_c, tstat_list, logp_list = eval_logrank(test_c, test_t, test_e)
    for c in range(args.k):
        print(
            "\tTest Cluster {}: Mean Survival Time {}, Censor Rate {}, Number {}".format(c, test_t[test_c == c].mean(),
                                                                                         test_e[test_c == c].mean(),
                                                                                         (test_c == c).sum()))
    print("\tAverage Test: tstats {}, logp {}".format(np.array(tstat_list).mean(), np.array(logp_list).mean()))
    print("{},{},{},{}".format(train_c_idx[0], test_c_idx[0], np.array(tstat_list).mean(), np.array(logp_list).mean()))

    print("Prediction C Index: Train {}, Test {}".format(train_c_idx[0], test_c_idx[0]))
    print("{},{}".format(train_c_idx[0], test_c_idx[0]))
    print("Final: {},{},{},{}".format(train_c_idx[0], test_c_idx[0], np.array(tstat_list).mean(), np.array(logp_list).mean()))

