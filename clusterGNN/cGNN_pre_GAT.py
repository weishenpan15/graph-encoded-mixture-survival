import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import numbers
import itertools
# from umap.umap_ import find_ab_params
import dill
# from umap import UMAP
from nsc import NeuralSurvivalCluster
from nsc.nsc_torch import NeuralSurvivalClusterTorchJ, NeuralSurvivalClusterTorch
from scipy.optimize import curve_fit

from torch_geometric.nn import GATConv, Linear, SuperGATConv, GINConv

import torch
import torch.nn.functional as F
from umap.umap_ import fuzzy_simplicial_set

# import dgl


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


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


# -------------------
def mask_edge(edge_index, mask_prob):
    E = edge_index.shape[-1]

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    # print(mask_rates.shape, mask_rates)
    masks = torch.bernoulli(1 - mask_rates)

    mask_idx = masks.nonzero().squeeze(1)

    nsrc = edge_index[0][mask_idx]
    ndst = edge_index[1][mask_idx]
    edegs = torch.stack((nsrc,ndst),0 ).reshape(2, -1)
    # print(nsrc.shape, ndst.shape, torch.stack((nsrc,ndst),0 ).shape)
    return edegs


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss


class GATModel1L(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATModel1L, self).__init__()
        self.conv = GATConv(input_dim, output_dim, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
        # x = F.prelu(x)

        return x


class GATModel1Ldp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATModel1Ldp, self).__init__()
        self.conv = GATConv(input_dim, output_dim, heads=1)

    def forward(self, edge_index, x, trainflag=False):
        # not using dropout actually
        # x = F.dropout(x, 0.1, training=trainflag)
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)

        return x


class GATModel2L(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATModel2L, self).__init__()
        self.conv = GATConv(input_dim, output_dim, heads=1)
        self.conv2 = GATConv(output_dim, input_dim, heads=1)

    def forward(self, data, trainflag):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, 0.1, training=trainflag)
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
        midh = x
        x = F.dropout(x, 0.1, training=trainflag)
        decodex = self.conv2(x, edge_index)
        decodex = F.leaky_relu(decodex)
        return midh, decodex


def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


def convert_distance_to_probability(distances, a=1.0, b=1.0):
    return -torch.log1p(a * distances ** (2 * b))


def convert_distance_to_probability1(distances):
    return -distances


def convert_distance_to_probability2(distances):
    probabilities = torch.exp(-distances)
    return probabilities


class Datamodule(pl.LightningDataModule):
    def __init__(
            self,
            dataset,
            batch_size,
            num_workers,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )


class GNNBasicLearner(torch.nn.Module):
    def __init__(
            self,
            lr: float,
            encoder: nn.Module,
            nsc,
            decoder=None,
            beta=1.0,
            # min_dist=0.1,
            reconstruction_loss=F.binary_cross_entropy_with_logits,
            # match_nonparametric_umap=False,
    ):
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        # self.k_centroid = k_centroid
        self.nsc = nsc
        self.beta = beta  # weight for reconstruction loss
        # self.match_nonparametric_umap = match_nonparametric_umap
        self.reconstruction_loss = reconstruction_loss
        # self._a, self._b = find_ab_params(1.0, min_dist)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.001)


class ClusterAttentionAE(torch.nn.Module):
    def __init__(
            self,
            k=2,
            metric="euclidean",
            reconstruction_loss=F.cosine_similarity,
            random_state=None,
            lr=0.005,
            epochs=10,
            batch_size=32,
            num_workers=0,
            num_gpus=1,
            graph=None,
            prek=None,
            modify_nsc=0,
            para_dict=None,
    ):
        super(ClusterAttentionAE, self).__init__()
        torch.manual_seed(random_state)

        self.cens = k
        assert self.cens == 2
        self.metric = metric
        self.reconstruction_loss = reconstruction_loss
        self.random_state = random_state
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.input_dim = graph.x.shape[-1]

        self.hidden_dim = 8
        self.output_dim = 8
        self.modify_nsc = modify_nsc
        self.encoder = GATModel1Ldp(self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim)
        self.decoder = GATModel1Ldp(self.output_dim, hidden_dim=self.input_dim, output_dim=self.input_dim)
        self.encoder_to_decoder = nn.Linear(self.output_dim, self.output_dim, bias=False)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.input_dim))
        self._decoder_type = 'gat'
        # self.dp = nn.Dropout(p=0.0)
        # hyperparameters
        self._replace_rate = 0.0
        self._mask_token_rate = 1 - self._replace_rate
        self._mask_rate = 0.1
        self._drop_edge_rate = 0.0
        if prek:
            self.k_centroid = prek
        else:
            self.k_centroid = nn.ParameterList(
                [nn.Parameter(torch.rand(self.output_dim, )) for k_idx in range(self.cens)])

        if self.modify_nsc:
            if para_dict is None:
                self.nsc = NeuralSurvivalClusterTorchJ(inputdim=self.output_dim, k=self.cens)
            else:
                self.nsc = NeuralSurvivalClusterTorchJ(inputdim=self.output_dim, k=self.cens,
                                                       layers=para_dict['layers'], act=para_dict['act'], layers_surv=para_dict['layers_surv'], representation=para_dict['representation'], act_surv=para_dict['act'])
        else:
            if para_dict is None:
                self.nsc = NeuralSurvivalClusterTorch(inputdim=self.output_dim, k=self.cens)
            else:
                self.nsc = NeuralSurvivalClusterTorch(inputdim=self.output_dim, k=self.cens,
                                                       layers=para_dict['layers'], act=para_dict['act'], layers_surv=para_dict['layers_surv'], representation=para_dict['representation'], act_surv=para_dict['act'])

        # self.reconstruction_loss = reconstruction_loss
        params = [self.encoder.parameters(), self.decoder.parameters(), self.encoder_to_decoder.parameters(), self.nsc.parameters(), self.k_centroid,  [self.enc_mask_token]]
        self.opt = torch.optim.AdamW(itertools.chain(*params), lr=self.lr)
        self.opt_pre = torch.optim.AdamW(list(self.encoder.parameters())+list(self.decoder.parameters()) + list(self.encoder_to_decoder.parameters()) + [self.enc_mask_token], lr=self.lr)

        min_dist = 0.1
        self._a, self._b = find_ab_params(1.0, min_dist)

    def encoding_mask_noise(self, edges, x, mask_rate=0.3):

        num_nodes = x.shape[0]

        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = edges.clone()
        return use_g, out_x, (mask_nodes, keep_nodes)

    def mask_attr_prediction(self, g, x):
        # x is the original x, use_x is the cloned and masked x
        # g is the original g, pre_use_g is the cloned g
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g = mask_edge(pre_use_g, self._drop_edge_rate)
            # : drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g  # cloned g

        enc_rep = self.encoder(use_g, use_x, trainflag=True)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        # rep = enc_rep
        # self._decoder_type is gat
        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "linear") :
            recon = self.decoder(rep)
        else:
            # pre_use_g.x = rep
            recon = self.decoder(pre_use_g, rep, trainflag=True)

        x_init = x[mask_nodes]  # comparison to the original x
        x_rec = recon[mask_nodes]
        loss = 1 - F.cosine_similarity(x_rec, x_init, dim=-1).mean()
        # loss = sce_loss(x_init, x_rec)
        return loss, enc_rep

    def get_loss(self, graph, batch, train_idx, weight_balance=1.0):
        # this is for pretrain loss
        (x, edges, train_t, train_e) = graph.x, graph.edge_index, graph.yt, graph.ye

        all_x_prop, decodex = self.model(graph, trainflag=True)

        train_idx_x_prop = all_x_prop[train_idx]
        train_idx_x_rec = decodex[train_idx]
        assert batch is None
        # here using cosine loss
        loss = 1 - F.cosine_similarity(x[train_idx], train_idx_x_rec, dim=-1).mean()

        return loss, None

    def fit(self, graph, train_idx, batch_size=32, epoch=200):
        # this is for pretrain
        train_num = len(train_idx)
        for i in range(epoch):
            dataloader = DataLoader(range(train_num), batch_size=batch_size, shuffle=True)

            # for batch in dataloader:
            for batch in range(1):

                self.opt_pre.zero_grad()
                # self.opt_clu.zero_grad()
                loss_pred , _ = self.mask_attr_prediction(graph.edge_index, graph.x)
                # loss_pred, _ = self.get_loss(graph, None, train_idx)
                loss_pred.backward()
                # for i, j in self.encoder.named_parameters():
                #     print(i, j.grad, j.shape)
                # for i, j in self.decoder.named_parameters():
                #     print(i, j.grad, j.shape)
                # print('token', self.enc_mask_token.grad, self.enc_mask_token)
                # print('k', self.k_centroid[0], self.k_centroid[0].grad)
                if i % 10 == 0 or i == epoch -1:
                    print(i, loss_pred)
                self.opt_pre.step()

    def get_all_loss(self, graph, batch, train_idx, weight_balance=1.0):
        # this is for all joint training loss
        (x, edges, train_t, train_e) = graph.x, graph.edge_index, graph.yt, graph.ye

        loss_rec, all_x_prop = self.mask_attr_prediction(edges, x)

        train_idx_x_prop = all_x_prop[train_idx]
        assert batch is None

        if self.modify_nsc:
            alphas = torch.cat([convert_distance_to_probability((train_idx_x_prop - self.k_centroid[k_idx]).norm(dim=1),
                                                                self._a, self._b).unsqueeze(1)
                                for k_idx in range(len(self.k_centroid))], axis=1)
            cumulative, intensity = self.nsc.forward(train_idx_x_prop, train_t, gradient=True)
        else:
            cumulative, intensity, alphas = self.nsc.forward(train_idx_x_prop, train_t, gradient=True)

        with torch.no_grad():
            intensity.clamp_(1e-10)

        alphas = nn.LogSoftmax(dim=1)(alphas)

        cum = alphas - cumulative.sum(1)  # Sum over all risks

        error = - weight_balance * torch.logsumexp(cum[train_e == 0],
                                                   dim=1).sum()  # Sum over the different mixture and then across patient

        for k in range(1):
            i = intensity[train_e == (k + 1)][:, k]
            error -= torch.logsumexp(cum[train_e == (k + 1)] + torch.log(i), dim=1).sum()

        loss_pred = error / len(train_t)

        return loss_pred, loss_rec

    def allfit(self, graph, train_idx, batch_size=32,         epoch = 200):
        # this is for joint training
        train_num = len(train_idx)
        for i in range(epoch):
            dataloader = DataLoader(range(train_num), batch_size=batch_size, shuffle=True)

            # for batch in dataloader:
            for batch in range(1):

                self.opt.zero_grad()
                # self.opt_clu.zero_grad()

                loss_pred, loss_rec = self.get_all_loss(graph, None, train_idx)
                (loss_pred + 0.005 * loss_rec).backward()
                # print('k', self.k_centroid[0], self.k_centroid[0].grad)

                self.opt.step()

                self.opt.zero_grad()
                self.opt_clu.zero_grad()
                _, loss_sign = self.get_loss(graph, batch, train_idx)
                loss_sign.backward()
                self.opt_clu.step()

    @torch.no_grad()
    def transform(self, X):
        print(f"Reducing array of shape {X.shape} to ({X.shape[0]}, {self.n_components})")
        return self.model.encoder(X).detach().cpu().numpy()

    # @torch.no_grad()
    def predict_survival(self, graph, index, tune=True):
        print('survival graph', graph)

        if tune:
            # now do not need tune
            # ae_params = [self.model.parameters(), self.decoder.parameters(), self.k_centroid]
            # ae_opt = torch.optim.AdamW(itertools.chain(*ae_params), lr=self.lr)
            for i in range(50):
                loss_rec, all_x_prop = self.mask_attr_prediction(graph.edge_index, graph.x)
                # all_x_prop = self.dp(all_x_prop)
                # rec_x = self.decoder(all_x_prop)
                # rec_x = F.leaky_relu(rec_x)
                self.opt_pre.zero_grad()
                # ae_opt.zero_grad()
                # loss_pred = F.mse_loss(graph.x[index], rec_x)
                # print('tune', i, loss_pred)

                print(loss_rec)
                loss_rec.backward()
                self.opt_pre.step()
            loss_rec, all_x_prop = self.mask_attr_prediction(graph.edge_index, graph.x)
            all_x_prop = all_x_prop[index]

        else:
            # loss_rec, all_x_prop = self.mask_attr_prediction(graph.edge_index, graph.x)
            all_x_prop = self.encoder(graph.edge_index, graph.x, trainflag=False)
            # all_x_prop = self.encoder_to_decoder(all_x_prop)

            all_x_prop = all_x_prop[index]

        if self.modify_nsc:
            alphas = torch.cat(
                [convert_distance_to_probability((all_x_prop - self.k_centroid[k_idx]).norm(dim=1)).unsqueeze(1)
                 for k_idx in range(len(self.k_centroid))], axis=1)
            alphas = nn.Softmax(dim=1)(alphas).unsqueeze(1).repeat(1, 1, 1)

        scores = []
        for t_ in [0.25, 0.5, 0.75]:
            t_ = torch.DoubleTensor([t_] * len(all_x_prop)).to(all_x_prop.device)
            if self.modify_nsc:
                outcomes, _ = self.nsc.predict(all_x_prop, alphas, t_)
            else:
                outcomes, alphas, _ = self.nsc.predict(all_x_prop, t_.float())

            scores.append(outcomes[:, int(1) - 1].unsqueeze(1).detach().cpu().numpy())
        return np.concatenate(scores, axis=1), alphas

    @torch.no_grad()
    def inverse_transform(self, Z):
        return self.model.decoder(Z).detach().cpu().numpy()

    def save(self, path):
        with open(path, 'wb') as oup:
            dill.dump(self, oup)
        print(f"Pickled PUMAP object at {path}")


class ClusterGNN(torch.nn.Module):
    def __init__(
            self,
            k=2,
            metric="euclidean",
            reconstruction_loss=F.binary_cross_entropy_with_logits,
            random_state=None,
            lr=0.001,
            epochs=10,
            batch_size=32,
            num_workers=0,
            num_gpus=1,
            graph=None,
            para_dict=None,
    ):
        super(ClusterGNN, self).__init__()
        torch.manual_seed(random_state)

        self.cens = k
        self.metric = metric
        self.reconstruction_loss = reconstruction_loss
        self.random_state = random_state
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.input_dim = graph.x.shape[-1]

        self.hidden_dim = 8
        self.output_dim = 8
        # self.b = nn.Parameter(torch.randn(1,))
        self.model = GATModel1L(self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim)

        if para_dict is None:
            self.nsc = NeuralSurvivalClusterTorch(inputdim=self.output_dim, k=self.cens)
        else:
            self.nsc = NeuralSurvivalClusterTorch(inputdim=self.output_dim, k=self.cens,
                                                   layers=para_dict['layers'], act=para_dict['act'], layers_surv=para_dict['layers_surv'], representation=para_dict['representation'], act_surv=para_dict['act'])

        self.reconstruction_loss = reconstruction_loss
        params = [self.model.parameters(), self.nsc.parameters()]
        self.opt = torch.optim.AdamW(itertools.chain(*params), lr=self.lr)
        # self.a, self.b = 0, 1

        params_clu = [self.model.parameters(), self.nsc.profile.parameters()]
        self.opt_clu = torch.optim.AdamW(itertools.chain(*params_clu), lr=self.lr)

        min_dist = 0.1
        self._a, self._b = find_ab_params(1.0, min_dist)

    def get_loss(self, graph, batch, train_idx, weight_balance=1.0):

        (x, edges, train_t, train_e) = graph.x, graph.edge_index, graph.yt, graph.ye

        all_x_prop = self.model(graph)

        train_idx_x_prop = all_x_prop[train_idx]

        if batch is not None:
            train_idx_x_prop = train_idx_x_prop[batch]
            train_t = train_t[batch]
            train_e = train_e[batch]

        cumulative, intensity, alphas = self.nsc.forward(train_idx_x_prop, train_t, gradient=True)

        with torch.no_grad():
            intensity.clamp_(1e-10)

        alphas = nn.LogSoftmax(dim=1)(alphas)

        cum = alphas - cumulative.sum(1)  # Sum over all risks

        error = - weight_balance * torch.logsumexp(cum[train_e == 0],
                                                   dim=1).sum()  # Sum over the different mixture and then across patient

        for k in range(1):
            i = intensity[train_e == (k + 1)][:, k]
            error -= torch.logsumexp(cum[train_e == (k + 1)] + torch.log(i), dim=1).sum()

        loss_pred = error / len(train_t)

        all_alphas = nn.Softmax(dim=1)(alphas)
        mod_graph = graph.mod_graph
        tr_mod = torch.matmul(all_alphas.transpose(1, 0), mod_graph)
        tr_mod = torch.matmul(tr_mod, all_alphas)
        loss_mod = -torch.trace(tr_mod) / edges.size()[1]

        n_cluster = alphas.size()[1]
        loss_cnt = 0
        for c1 in range(n_cluster):
            for c2 in range(c1 + 1, n_cluster):
                switch_idx = [tmp_idx for tmp_idx in range(n_cluster)]
                switch_idx[c1] = c2
                switch_idx[c2] = c1
                # print(switch_idx)
                alphas_switch = alphas[:, torch.tensor(switch_idx)]
                cum = alphas_switch - cumulative.sum(1)  # Sum over all risks
                error = - weight_balance * torch.logsumexp(cum[train_e == 0],
                                                           dim=1).sum()  # Sum over the different mixture and then across patient
                for k in range(1):
                    i = intensity[train_e == (k + 1)][:, k]
                    error -= torch.logsumexp(cum[train_e == (k + 1)] + torch.log(i), dim=1).sum()

                if c1 == 0 and c2 == 1:
                    loss_cross = error / len(train_t)
                else:
                    loss_cross += error / len(train_t)

                loss_cnt += 1
        loss_cross = loss_cross / loss_cnt
        loss_balance = torch.std(torch.exp(alphas).mean(dim=0))
        loss_sign = loss_pred - loss_cross

        if n_cluster > 2:
            loss = loss_pred + loss_balance
        else:
            loss = loss_pred

        return loss, loss_sign, loss_mod

    def fit(self, graph, train_idx):
        train_num = len(train_idx)
        for i in range(self.epochs):
            # for batch in dataloader:
            for batch in range(1):
                self.opt.zero_grad()
                self.opt_clu.zero_grad()
                loss_pred, _, _ = self.get_loss(graph, None, train_idx)
                loss_pred.backward()
                self.opt.step()
                self.opt_clu.step()

                self.opt.zero_grad()
                self.opt_clu.zero_grad()
                _, loss_sign, loss_mod = self.get_loss(graph, None, train_idx)
                loss = 0.01 * loss_mod + 0.01 * loss_sign
                loss.backward()
                self.opt_clu.step()

    @torch.no_grad()
    def transform(self, X):
        print(f"Reducing array of shape {X.shape} to ({X.shape[0]}, {self.n_components})")
        return self.model.encoder(X).detach().cpu().numpy()

    @torch.no_grad()
    def predict_survival(self, graph, index):
        print('survival graph', graph)
        all_x_prop = self.model(graph)
        all_x_prop = all_x_prop[index]

        scores = []
        for t_ in [0.25, 0.5, 0.75]:
            t_ = torch.DoubleTensor([t_] * len(all_x_prop)).to(all_x_prop.device)
            outcomes, alphas, _ = self.nsc.predict(all_x_prop, t_.float())

            scores.append(outcomes[:, int(1) - 1].unsqueeze(1).detach().cpu().numpy())
        return np.concatenate(scores, axis=1), alphas

    @torch.no_grad()
    def inverse_transform(self, Z):
        return self.model.decoder(Z).detach().cpu().numpy()

    def save(self, path):
        with open(path, 'wb') as oup:
            dill.dump(self, oup)
        print(f"Pickled PUMAP object at {path}")


if __name__ == "__main__":
    pass
