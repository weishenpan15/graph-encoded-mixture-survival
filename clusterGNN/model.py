import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class conv_encoder(nn.Module):
    def __init__(self, n_components=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1,
            ),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1,
            ),
            nn.Flatten(),
            nn.Linear(6272, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_components)
        ).cuda()
    def forward(self, X):
        return self.encoder(X)
   
# class default_encoder(nn.Module):
#     def __init__(self, dims, n_components=2):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(np.product(dims), 200),
#             nn.ReLU(),
#             nn.Linear(200,200),
#             nn.ReLU(),
#             nn.Linear(200,200),
#             nn.ReLU(),
#             nn.Linear(200, n_components),
#         ).cuda()
#
#     def forward(self, X):
#         return self.encoder(X)


class default_encoder(nn.Module):
    def __init__(self, dims, n_components=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dims[0], 8),
            nn.ReLU(),
            nn.Linear(8, n_components),
        )
        # self.encoder = nn.Sequential(
        #     nn.Linear(dims[0], 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, n_components),
        # )

    def forward(self, X):
        return self.encoder(X)

# class default_decoder(nn.Module):
#     def __init__(self, dims, n_components):
#         super().__init__()
#         self.dims = dims
#         self.decoder = nn.Sequential(
#             nn.Linear(n_components, 200),
#             nn.ReLU(),
#             nn.Linear(200,200),
#             nn.ReLU(),
#             nn.Linear(200,200),
#             nn.ReLU(),
#             nn.Linear(200, np.product(dims)),
#         ).cuda()
#     def forward(self, X):
#         return self.decoder(X).view(X.shape[0], *self.dims)

class default_decoder(nn.Module):
    def __init__(self, dims, n_components):
        super().__init__()
        self.dims = dims
        self.decoder = nn.Sequential(
            nn.Linear(n_components, 200),
            nn.ReLU(),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200, np.product(dims)),
        )
    def forward(self, X):
        return self.decoder(X).view(X.shape[0], *self.dims)
   

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv


# Define your GNN model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        # print(data)
        x, edge_index, yt, ye = data.x, data.edge_index, data.yt, data.ye
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        # batchx = x[index]
        # batchyt = yt[index]
        # batchye = ye[index]

        return x

class GNNModel1L(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super(GNNModel1L, self).__init__()
        self.conv = GCNConv(input_dim, hidden_dim)

    def forward(self, data):
        # print(data)
        x, edge_index = data.x, data.edge_index
        # print(x.shape, edge_index.shape)
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
        # x = self.conv2(x, edge_index)

        # batchx = x[index]
        # batchyt = yt[index]
        # batchye = ye[index]

        return x




if __name__ == "__main__":
    model = conv_encoder(2)
    print(model.parameters)
    print(model(torch.randn((12,1,28,28)).cuda()).shape)


