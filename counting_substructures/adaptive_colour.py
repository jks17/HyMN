import torch
import torch.nn.functional as F

import torch_scatter

class AdaptiveColouring(object):

    def __init__(self, seed, device, num_samples, ratio, *args, **kwargs):

        self.seed = seed
        self.device = device
        self.num_samples = num_samples
        self.ratio = ratio
        self.generator = torch.Generator(device)
        self.reset()

    def _get_sizes(self, batch):
        return torch.diff(batch.ptr)

    def _get_auxiliary_indices(self, g, graph_size, inc):
        valid = int(graph_size * self.ratio)
        assert valid <= self.num_samples
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
            indices = torch.zeros([self.num_samples], device=self.device, dtype=torch.int64)
            indices[:lowest] = torch.randperm(int(graph_size), generator=self.generator, device=self.device)[:lowest]
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
            indices = torch.zeros([self.num_samples], device=self.device, dtype=torch.int64)
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


def build_adaptive_painter(colouring_scheme, seed, device, num_samples, ratio, *args, **kwargs):
    if colouring_scheme == 'index':
        return AdaptiveIndexColouring(seed, device, num_samples, ratio, *args, **kwargs)
    if colouring_scheme in ['indexcommunity', 'indextmd', 'indexrandom']:
        return AdaptiveIndexColouringCommunity(seed, device, num_samples, ratio, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported scheme '{colouring_scheme}'")
