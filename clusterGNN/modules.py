from pynndescent import NNDescent
import numpy as np
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set
import torch

def convert_distance_to_probability(distances, a=1.0, b=1.0):
    return -torch.log1p(a * distances ** (2 * b))

def compute_cross_entropy(
    probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0
):
    # cross entropy
    # print('in ', probabilities_graph.device, probabilities_distance.device)
    attraction_term = -probabilities_graph * torch.nn.functional.logsigmoid(
        probabilities_distance
    )
    repellant_term = (
        -(1.0 - probabilities_graph)
        * (torch.nn.functional.logsigmoid(probabilities_distance)-probabilities_distance)
        * repulsion_strength)

    # balance the expected losses between atrraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE

def umap_loss(embedding_to, embedding_from, _a, _b, batch_size, negative_sample_rate=5):
    # get negative samples by randomly shuffling the batch
    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]
    distance_embedding = torch.cat((
        (embedding_to - embedding_from).norm(dim=1),
        (embedding_neg_to - embedding_neg_from).norm(dim=1)
    ), dim=0)

    # convert probabilities to distances
    probabilities_distance = convert_distance_to_probability(
        distance_embedding, _a, _b
    )
    # set true probabilities based on negative sampling
    probabilities_graph = torch.cat(
        (torch.ones(batch_size), torch.zeros(batch_size * negative_sample_rate)), dim=0,
    )

    # compute cross entropy
    (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
        probabilities_graph.to(probabilities_distance.device),
        probabilities_distance,
    )
    loss = torch.mean(ce_loss)
    return loss

def umap_loss_weight(embedding_to, embedding_from, e, _a, _b, batch_size, negative_sample_rate=5, w = 1.394):
    # get negative samples by randomly shuffling the batch

    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]
    distance_embedding = torch.cat((
        (embedding_to - embedding_from).norm(dim=1),
        (embedding_neg_to - embedding_neg_from).norm(dim=1)
    ), dim=0)

    weight_to = 1 * e + w * (1 - e)
    repeat_weight_to = weight_to.repeat(negative_sample_rate + 1)
    # convert probabilities to distances
    probabilities_distance = convert_distance_to_probability(
        distance_embedding, _a, _b
    )
    # set true probabilities based on negative sampling
    probabilities_graph = torch.cat(
        (torch.ones(batch_size), torch.zeros(batch_size * negative_sample_rate)), dim=0,
    )

    # compute cross entropy
    # print(probabilities_graph.device,probabilities_distance.device )

    (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
        probabilities_graph.to(probabilities_distance.device),
        probabilities_distance,
    )
    loss = (ce_loss * repeat_weight_to).sum() / repeat_weight_to.sum()
    # loss = torch.mean(ce_loss)
    return loss

def get_umap_graph(X, t, e, dist_dict, n_neighbors=10, random_state=None):
    random_state = check_random_state(None) if random_state == None else random_state

    t_list = dist_dict['t_list']

    dis_t_c2c = dist_dict['dis_t_c2c']
    dis_t_u2u = dist_dict['dis_t_u2u']
    dis_t_u2c = dist_dict['dis_t_u2c']

    t_map = {t_list[idx]:idx for idx in range(len(t_list))}
    dmat = np.zeros((X.shape[0], X.shape[0]))
    for i_idx in range(dmat.shape[0]):
        for j_idx in range(dmat.shape[1]):
            t_i = max(1, int(np.floor(t[i_idx])))
            t_j = max(1, int(np.floor(t[j_idx])))
            t_i_idx = t_map[t_i]
            t_j_idx = t_map[t_j]


            # dmat[i_idx, j_idx] = np.abs(t_i - t_j)
            # dmat[i_idx, j_idx] = np.abs(np.log(t_i / t_j))

            if e[i_idx] == 1 and e[j_idx] == 1:
                dmat[i_idx, j_idx] = dis_t_u2u[t_i_idx, t_j_idx]
            elif e[i_idx] == 0 and e[j_idx] == 0:
                dmat[i_idx, j_idx] = dis_t_c2c[t_i_idx, t_j_idx]
            elif e[i_idx] == 1 and e[j_idx] == 0:
                dmat[i_idx, j_idx] = dis_t_u2c[t_i_idx, t_j_idx]
            elif e[i_idx] == 0 and e[j_idx] == 1:
                dmat[i_idx, j_idx] = dis_t_u2c[t_j_idx, t_i_idx]

    # build fuzzy_simplicial_set
    umap_graph, sigmas, rhos = fuzzy_simplicial_set(
        X = dmat,
        n_neighbors = n_neighbors,
        metric = "precomputed",
        random_state = random_state,
        knn_indices= None,
        knn_dists = None,
    )
    
    return umap_graph