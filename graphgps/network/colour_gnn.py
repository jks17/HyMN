import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer, GINEConvLayerColour
from graphgps.layer.gcn_conv_layer import GCNConvLayerColour
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from graphgps.layer.nn import SharedLinear, SharedMLP
from graphgps.network.readout import (SumReadout, MeanReadout, MaxReadout, SumJKReadout, MeanJKReadout, MaxJKReadout,
    MeanAveraging, SumAveraging, MaxAveraging, SharedJKReadout, ColourReadout, ColourJKReadout,
    AdaptiveMeanAveraging, AdaptiveSumAveraging, AdaptiveMaxAveraging)

import torch
import torch.nn.functional as F

import torch_scatter


@register_network('colour_gnn')
class ColourGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        num_samples = cfg.gnn.num_samples
        self.jk = cfg.gnn.jk
        
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(conv_model(dim_in,
                                     dim_in,
                                     1, num_samples,   # colour_dim, num_samples
                                     dropout=cfg.gnn.dropout,
                                     norm=cfg.gnn.batchnorm,
                                     residual=cfg.gnn.residual,
                                     jk=cfg.gnn.jk))
        self.gnn_layers = torch.nn.Sequential(*layers)
        
        if cfg.gnn.subgraph_pooling == 'mean':
            self.readout_module = MeanAveraging(dim_in, num_samples)
        elif cfg.gnn.subgraph_pooling == 'sum':
            self.readout_module = SumAveraging(dim_in, num_samples)
            
        self.num_samples = num_samples
        self.emb_dim = dim_in


        GNNHead = register.head_dict[cfg.gnn.head]
        
        if cfg.gnn.jk:
            self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner * (cfg.gnn.layers_mp), dim_out=dim_out)
        else:
            self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'gineconv':
            return GINEConvLayerColour
        elif model_type == 'gcnconv':
            return GCNConvLayerColour
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        batch = self.encoder(batch)
        batch.x_repeated = batch.x.repeat((1,self.num_samples))
        if batch.edge_attr is not None:
            if batch.edge_attr.ndim == 1:
                batch.edge_attr_repeated = batch.edge_attr.unsqueeze(1).repeat((1,self.num_samples))
            else:
                batch.edge_attr_repeated = batch.edge_attr.repeat((1,self.num_samples))
                
        batch = self.gnn_layers(batch)  # batch.x_repeated is of shape (num_nodesxnum_samples) x emb_dim
        batch.x = batch.x_repeated
        
        batch.x = self.readout_module(batch.x_repeated) # this averages over num_samples, batch.sample_index, batch.sample_batch)
        batch = self.post_mp(batch)
        
        return batch
