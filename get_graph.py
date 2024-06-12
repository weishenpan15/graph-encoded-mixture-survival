import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(0)
def knn_search(query, data, k=5, debug=False):
    assert k <= len(data)
    dists = np.sqrt(np.sum((data - query) ** 2, axis=1))  # euclidean distance
    if debug:
        print("[DEBUG] max dist =", np.max(dists))
        print("[DEBUG] min dist =", np.min(dists))
        print("[DEBUG] mean dist =", np.mean(dists))
    inds = np.argsort(dists)  # sorted in ascending order
    inds_k = inds[:k]         # top k closest data points
    # NOTE: optionally, if the argumet dataset has a set of labels, we can also
    # associate the query vector with a label (i.e., classification).
    return inds_k[:], dists[inds_k]

def hamming_hash(data, hyperplanes):
    b = len(hyperplanes)
    hash_key = (data @ hyperplanes.T) >= 0
    dec_vals = np.array([2 ** i for i in range(b)], dtype=int)
    hash_key = hash_key @ dec_vals
    return hash_key

def generate_hyperplanes(data, bucket_size=16):
    m = data.shape[0]            # number of data points
    n = data.shape[1]            # number of features in a data point
    b = m // bucket_size         # desired number of hash buckets
    h = int(np.log2(b))          # desired number of hyperplanes
    H = rng.normal(size=(h, n))  # hyperplanes as their normal vectors
    return H

def locality_sensitive_hash(data, hyperplanes):
    hash_vals = hamming_hash(data, hyperplanes)
    hash_table = {}
    for i, v in enumerate(hash_vals):
        if v not in hash_table:
            hash_table[v] = set()
        hash_table[v].add(i)
    return hash_table

def approx_knn_search(query, data, k=5, bucket_size=16, repeat=10, debug=False):
    candidates_idx = set()
    for i in range(repeat):
        hyperplanes = generate_hyperplanes(data)
        hash_table = locality_sensitive_hash(data, hyperplanes)
        if debug:
            avg_bucket_size = np.mean([len(v) for v in hash_table.values()])
            print(f"[DEBUG] i = {i}, avg_bucket_size = {avg_bucket_size}")
        query_hash = hamming_hash(query, hyperplanes)
        if query_hash in hash_table:
            candidates_idx = candidates_idx.union(hash_table[query_hash])
    candidates_idx = list(candidates_idx)
    candidates = np.stack([data[i] for i in candidates_idx], axis=0)
    tmp_idx, tmp_dist = knn_search(query, candidates, k=k, debug=debug)
    return np.array([candidates_idx[idx] for idx in tmp_idx]), tmp_dist

def get_graph(kcore, patient_embed, patient_features, target_t, target_e):
    k_neighbors = kcore
    # knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    #
    # knn_model.fit(patient_embed)
    # knn_distances, knn_indices = knn_model.kneighbors(patient_embed, n_neighbors=k_neighbors)

    knn_distances, knn_indices = [], []
    for idx in range(patient_embed.shape[0]):
        tmp_knn_indices, tmp_knn_distances  = approx_knn_search(patient_embed[idx, :], patient_embed, k=k_neighbors)
        knn_distances.append(tmp_knn_distances)
        knn_indices.append(tmp_knn_indices)
    knn_indices, knn_distances = np.array(knn_indices), np.array(knn_distances)

    num_patients = patient_features.shape[0]
    graphadj = np.zeros((num_patients, num_patients), dtype=float)

    for i in range(num_patients):
        for j_idx, j in enumerate(knn_indices[i]):
            dist_ij = knn_distances[i, j_idx]
            dist_i = knn_distances[i, :].mean()
            dist_j = knn_distances[j, :].mean()
            dist_nor = (dist_ij + dist_i + dist_j) / 3
            graphadj[j, i] = np.exp(-(dist_ij ** 2) / (dist_nor * 10))

    edge_index = torch.tensor(np.argwhere(graphadj)).T
    x = torch.tensor(patient_features, dtype=torch.float)

    k_in = graphadj.sum(axis=0, keepdims=True)
    k_out = graphadj.sum(axis=1, keepdims=True)
    mod_graph = graphadj - np.matmul(k_out, k_in) / graphadj.sum()
    mod_graph = torch.tensor(mod_graph.astype(np.float32))

    yt = torch.tensor(target_t, dtype=torch.float)  # Assuming 'target' is a tensor of target values
    ye = torch.tensor(target_e, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, yt=yt, ye=ye, mod_graph = mod_graph)

    return data, graphadj

def get_test_graph(best_k, train_pat, train_graph, test_pat):
    knn_distances, knn_indices = [], []
    for idx in range(test_pat.shape[0]):
        tmp_knn_indices, tmp_knn_distances = approx_knn_search(test_pat[idx,:], train_pat, k=best_k)
        knn_distances.append(tmp_knn_distances)
        knn_indices.append(tmp_knn_indices)
    knn_indices, knn_distances = np.array(knn_indices), np.array(knn_distances)

    num_patients = train_graph.x.shape[0]
    add_patients = test_pat.shape[0]

    edge_i = []
    edge_j = []
    for i in range(add_patients):
        edge_i.append(i + num_patients)
        edge_j.append(i + num_patients)
        for j in knn_indices[i]:
            edge_i.append(j)
            edge_j.append(i + num_patients)

    edges = torch.tensor(list(zip(edge_i, edge_j))).T
    edge_indexes = torch.cat((train_graph.edge_index, edges), -1)
    x = torch.tensor(test_pat, dtype=torch.float)
    xx = torch.cat((train_graph.x, x), 0)
    data = Data(x=xx, edge_index=edge_indexes)
    return data