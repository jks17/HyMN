import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg

from torch_geometric.nn import MessagePassing

from graphgps.layer.nn import (reset, Reshaper, ColourCatter, SharedLinear, SharedMLP, 
    ColourCatSharedLinear, ColourCatSharedMLP, CodeqSharedMLP, CodeqSharedLinear)


class GINEConvESLapPE(pyg_nn.conv.MessagePassing):
    """GINEConv Layer with EquivStableLapPE implementation.

    Modified torch_geometric.nn.conv.GINEConv layer to perform message scaling
    according to equiv. stable PEG-layer with Laplacian Eigenmap (LapPE):
        ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
    """
    def __init__(self, nn, eps=0., train_eps=False, edge_dim=None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if hasattr(self.nn[0], 'in_features'):
                in_channels = self.nn[0].in_features
            else:
                in_channels = self.nn[0].in_channels
            self.lin = pyg_nn.Linear(edge_dim, in_channels)
        else:
            self.lin = None
        self.reset_parameters()

        if hasattr(self.nn[0], 'in_features'):
            out_dim = self.nn[0].out_features
        else:
            out_dim = self.nn[0].out_channels

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.mlp_r_ij = torch.nn.Sequential(
            torch.nn.Linear(1, out_dim), torch.nn.ReLU(),
            torch.nn.Linear(out_dim, 1),
            torch.nn.Sigmoid())

    def reset_parameters(self):
        pyg_nn.inits.reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()
        pyg_nn.inits.reset(self.mlp_r_ij)

    def forward(self, x, edge_index, edge_attr=None, pe_LapPE=None, size=None):
        # if isinstance(x, Tensor):
        #     x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             PE=pe_LapPE, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j, edge_attr, PE_i, PE_j):
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
        r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim

        return ((x_j + edge_attr).relu()) * r_ij

    def __repr__(self):
        return f'{self.__class__.__name__}(nn={self.nn})'


class GINEConvLayer(nn.Module):
    """Graph Isomorphism Network with Edge features (GINE) layer.
    """
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        gin_nn = nn.Sequential(
            pyg_nn.Linear(dim_in, dim_out), nn.ReLU(),
            pyg_nn.Linear(dim_out, dim_out))
        self.model = pyg_nn.GINEConv(gin_nn)

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)

        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch


class ColourCatSharedZINCGINConv(MessagePassing):
    '''
        GINE convolutional layer which runs on the concatenation of colours and features and on
        multiple colouring samples at the same time.
    '''

    def __init__(self, in_dim, emb_dim, colour_dim, num_samples, multiplier=1, bn=True, encode_input=False):
        super(ColourCatSharedZINCGINConv, self).__init__(aggr="add")
        if not encode_input:
            self.node_encoder = None
        else:
            self.node_encoder = torch.nn.Embedding(ZINC_NUM_ATOMS, emb_dim)
            in_dim = emb_dim
        self.emb_dim = emb_dim
        self.num_samples = num_samples
        self.colour_catter = ColourCatter(in_dim, colour_dim, num_samples, out_reshape=True)
        self.mlp = SharedMLP(in_dim+colour_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn)
        self.eps = torch.nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.bond_encoder = torch.nn.Embedding(ZINC_NUM_BONDS, in_dim)

    def reset_parameters(self):
        if self.node_encoder is not None:
            self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.colour_catter.reset_parameters()
        self.mlp.reset_parameters()
        self.eps.data.fill_(0.0)

    def forward(self, x, edge_index, c, edge_attr):
        if self.node_encoder is not None:
            x = self.node_encoder(x.squeeze())
            x = x.reshape(-1, self.num_samples*self.emb_dim)
        edge_embedding = self.bond_encoder(edge_attr.squeeze())
        edge_embedding = edge_embedding.reshape(-1, self.num_samples*self.emb_dim)
        msg = self.propagate(edge_index, x=x, edge_attr=edge_embedding, c=c)
        out = self.mlp((1 + self.eps) * self.colour_catter(x, c) + msg)
        return out

    def message(self, x_j, c_j, edge_attr):
        return self.colour_catter(torch.nn.functional.relu(x_j + edge_attr), c_j)

    def update(self, aggr_out):
        return aggr_out


class GINEConvLayerColour(MessagePassing):
    """Graph Isomorphism Network with Edge features (GINE) layer.
    """
    def __init__(self, dim_in, dim_out, colour_dim, num_samples, dropout, residual):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.colour_dim = colour_dim 
        self.num_samples = num_samples

        self.eps = torch.nn.Parameter(torch.Tensor([0]), requires_grad=True)


        self.colour_catter = ColourCatter(dim_in, colour_dim, num_samples, out_reshape=True)
        self.mlp = SharedMLP(dim_in+colour_dim, dim_in, num_samples, multiplier=1, bn=True)
        #self.model = pyg_nn.GINEConv(mlp)
        
        #gin_nn = nn.Sequential(
        #    pyg_nn.Linear(dim_in, dim_out), nn.ReLU(),
        #    pyg_nn.Linear(dim_out, dim_out))

        
        #self.model = pyg_nn.GINEConv(gin_nn)

    def forward(self, batch):
        #batch.x = self.colour_catter(batch.x_repeated, batch.c_samples)
        x_in = batch.x_repeated

        #batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr_repeated)
        msg = self.propagate(batch.edge_index, x=batch.x_repeated, edge_attr=batch.edge_attr_repeated, c=batch.c_samples)
        batch.x_repeated = self.mlp((1 + self.eps) * self.colour_catter(batch.x_repeated, batch.c_samples) + msg)
        #batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)

        batch.x_repeated = F.relu(batch.x_repeated)
        batch.x_repeated = F.dropout(batch.x_repeated, p=self.dropout, training=self.training)

        if self.residual:
            batch.x_repeated = x_in + batch.x_repeated  # residual connection

        return batch

    def message(self, x_j, c_j, edge_attr):
        return self.colour_catter(torch.nn.functional.relu(x_j + edge_attr), c_j)

    def update(self, aggr_out):
        return aggr_out


@register_layer('gineconv')
class GINEConvGraphGymLayer(nn.Module):
    """Graph Isomorphism Network with Edge features (GINE) layer.
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        gin_nn = nn.Sequential(
            Linear_pyg(layer_config.dim_in, layer_config.dim_out), nn.ReLU(),
            Linear_pyg(layer_config.dim_out, layer_config.dim_out))
        self.model = pyg_nn.GINEConv(gin_nn)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch
