import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer, GINEConvLayerColour
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from graphgps.layer.nn import SharedLinear, SharedMLP
from graphgps.network.readout import (SumReadout, MeanReadout, MaxReadout, SumJKReadout, MeanJKReadout, MaxJKReadout,
    MeanAveraging, SumAveraging, MaxAveraging, SharedJKReadout, ColourReadout, ColourJKReadout,
    AdaptiveMeanAveraging, AdaptiveSumAveraging, AdaptiveMaxAveraging)

import torch
import torch.nn.functional as F

import torch_scatter

class AdaptiveColouring(object):

    def __init__(self, seed, num_samples, ratio, *args, **kwargs):

        self.seed = seed
        self.num_samples = num_samples
        self.ratio = ratio
        self.generator = torch.Generator(cfg.devices)
        self.reset()

    def _get_sizes(self, batch):
        return torch.diff(batch.ptr)

    def _get_auxiliary_indices(self, g, graph_size, inc):
        valid = min(int(graph_size * self.ratio), self.num_samples)
        #assert valid <= self.num_samples
        index = torch.arange(valid) + inc
        batch_val = torch.full([valid], g)
        return index, batch_val
        
    def colours(self, batch):
        raise NotImplementedError

    def paint(self, batch):
        colours, sample_index, sample_batch = self.colours(batch)
        batch.x_repeated = batch.x.repeat((1,self.num_samples))
        if batch.edge_attr is not None:
            if batch.edge_attr.ndim == 1:
                batch.edge_attr_repeated = batch.edge_attr.unsqueeze(1).repeat((1,self.num_samples))
            else:
                batch.edge_attr_repeated = batch.edge_attr.repeat((1,self.num_samples))
        batch.c_samples = colours
        batch.sample_index = sample_index
        batch.sample_batch = sample_batch
        return batch

    def reset(self):
        self.generator = self.generator.manual_seed(self.seed)

class AdaptiveIndexColouring(AdaptiveColouring):

    def colours(self, batch):
        sizes = self._get_sizes(batch)
        inc = 0
        colours = list()
        sample_index = list()
        sample_batch = list()
        for g, graph_size in enumerate(sizes):
            index, batch_val = self._get_auxiliary_indices(g, graph_size, inc)
            lowest = min(graph_size, self.num_samples)
            indices = torch.zeros([self.num_samples], device=batch.x.device, dtype=torch.int64)
            indices[:lowest] = torch.randperm(int(graph_size), generator=self.generator, device=batch.x.device)[:lowest]
            coloring = F.one_hot(indices, num_classes=int(graph_size)).T
            colours.append(coloring)
            sample_index.append(index)
            sample_batch.append(batch_val)
            inc += self.num_samples
        colours = torch.cat(colours,0)
        sample_index = torch.cat(sample_index,0)
        sample_batch = torch.cat(sample_batch,0)
        return colours, sample_index, sample_batch

class AdaptiveIndexColouringCommunity(AdaptiveColouring):

    def colours(self, batch):
        sizes = self._get_sizes(batch)
        inc = 0
        xi = 0
        colours = list()
        sample_index = list()
        sample_batch = list()
        for g, graph_size in enumerate(sizes):
            index, batch_val = self._get_auxiliary_indices(g, graph_size, inc)
            lowest = min(graph_size, self.num_samples)
            indices = torch.zeros([self.num_samples], device=batch.x.device, dtype=torch.int64)
            centrality = batch.community[xi:xi+graph_size]
            indices[:lowest] = centrality[:lowest]
            coloring = F.one_hot(indices, num_classes=int(graph_size)).T
            colours.append(coloring)
            sample_index.append(index)
            sample_batch.append(batch_val)
            inc += self.num_samples
            xi += graph_size
        colours = torch.cat(colours,0)
        sample_index = torch.cat(sample_index,0)
        sample_batch = torch.cat(sample_batch,0)
        return colours, sample_index, sample_batch


@register_network('colour_gnn')
class ColourGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        num_samples = 5
        ratio = 1.0

        self.painter = AdaptiveIndexColouring(cfg.seed, num_samples, ratio)
        #self.painter = AdaptiveIndexColouringCommunity(cfg.seed, num_samples, ratio)
        
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
                                     residual=cfg.gnn.residual))
        self.gnn_layers = torch.nn.Sequential(*layers)

        #self.readout_module = MeanReadout()
        self.readout_module = MeanAveraging(dim_in, num_samples)

        #predictor_in_dim = dim_in
        #predictor_in_dim += 1
        #self.prediction_module = SharedMLP(predictor_in_dim, dim_out, num_samples)
        #self.averaging_module = AdaptiveMeanAveraging(dim_out)
        
        #AdaptiveMeanAveraging(dim_in)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'gineconv':
            return GINEConvLayerColour
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        batch = self.encoder(batch)
        batch = self.painter.paint(batch)
        #print(batch)
        #print(batch.c_samples[:20])
        #print(batch.x_repeated[0][:200])
        batch = self.gnn_layers(batch)

        #hs = self.readout_module(batch.x_repeated, batch.batch)
        #ys = self.prediction_module(hs)
        #y = self.averaging_module(ys, batch.sample_index, batch.sample_batch)
        #batch = y, batch.y

        #print(y)
        
        batch.x = self.readout_module(batch.x_repeated) #, batch.sample_index, batch.sample_batch)
        batch = self.post_mp(batch)
        
        #for module in self.children():
        #    print(module)
        #    batch = module(batch)
        return batch
