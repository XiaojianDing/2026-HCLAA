import torch.nn as nn
from torch.nn.functional import normalize
import torch


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


# Decoder
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


# SCMVC Network
class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, device):
        super(Network, self).__init__()
        self.view = view
        self.encoders = nn.ModuleList([Encoder(input_size[v], feature_dim).to(device) for v in range(view)])
        self.decoders = nn.ModuleList([Decoder(input_size[v], feature_dim).to(device) for v in range(view)])

        # global features fusion layer
        self.feature_fusion_module = nn.Sequential(
            nn.Linear(self.view * feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, high_feature_dim)
        )

        # view-consensus features learning layer
        self.common_information_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim)
        )

    def feature_fusion(self, zs, zs_gradient):
        H = torch.cat(zs, dim=1)
        H = self.feature_fusion_module(H)
        return normalize(H, dim=1)

    def compute_fusion_weight(self, z1, z2):
        sim1 = torch.norm(z1, dim=1, keepdim=True)
        sim2 = torch.norm(z2, dim=1, keepdim=True)
        weight1 = sim1 / (sim1 + sim2 + 1e-8)
        weight2 = sim2 / (sim1 + sim2 + 1e-8)
        return weight1, weight2

    def fuse_views(self, zs):
        new_zs = []
        for i in range(self.view):
            for j in range(i + 1, self.view):
                weight1, weight2 = self.compute_fusion_weight(zs[i], zs[j])
                new_z = weight1 * zs[i] + weight2 * zs[j]
                new_zs.append(new_z)
        return new_zs

    def forward(self, xs, zs_gradient=True):
        rs, xrs, zs = [], [], []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            r = normalize(self.common_information_module(z), dim=1)
            rs.append(r)
            zs.append(z)
            xrs.append(xr)

        new_zs = self.fuse_views(zs)
        new_rs = [normalize(self.common_information_module(new_z), dim=1) for new_z in new_zs]
        H = self.feature_fusion(zs, zs_gradient)


        return xrs, zs, rs, H,new_rs
