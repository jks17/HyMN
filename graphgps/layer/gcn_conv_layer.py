import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg

from torch_geometric.nn import MessagePassing 

from graphgps.layer.nn import (reset, Reshaper, ColourCatter, SharedLinear, SharedMLP, 
    ColourCatSharedLinear, ColourCatSharedMLP, CodeqSharedMLP, CodeqSharedLinear)


class GCNConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(self.dropout),
        )
        self.model = pyg_nn.GCNConv(dim_in, dim_out, bias=True)

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index)
        batch.x = self.act(batch.x)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch
    
    
class  GCNConvLayerColour(MessagePassing):
    def __init__(self, dim_in, dim_out, colour_dim, num_samples, dropout, norm, residual, jk):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.colour_dim = colour_dim 
        self.num_samples = num_samples
        self.norm=norm
        self.jk = jk
        
        if self.norm:
            self.in_resh = Reshaper(dim_out, num_samples, out=False, stacked_samples=True)
            self.out_resh = Reshaper(dim_out, num_samples, out=True, stacked_samples=True)
            self.batchnorm = nn.BatchNorm1d(dim_out)
            
        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(self.dropout),
        )
        
        self.lin = SharedLinear(dim_in+colour_dim, dim_in, num_samples)
        self.colour_catter = ColourCatter(dim_in, colour_dim, num_samples, out_reshape=True)

    def forward(self, batch):
        x_in = batch.x_repeated
        batch.x_repeated = self.lin(self.colour_catter(batch.x_repeated, batch.c_samples))
        batch.x_repeated = self.propagate(batch.edge_index, x=batch.x_repeated)
        batch.x_repeated = self.act(batch.x_repeated)
        
        
        if self.residual:
            batch.x_repeated = x_in + batch.x_repeated  # residual connection
            
        if self.norm:
            h = self.in_resh(batch.x_repeated)
            h = self.batchnorm(h)
            batch.x_repeated = self.out_resh(h)
            
        return batch


    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
    def message_and_aggregate(self, adj_t, x):
        return spmm(adj_t, x, reduce=self.aggr)

    def update(self, aggr_out):
        return aggr_out