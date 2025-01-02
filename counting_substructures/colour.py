import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
import torch_scatter


class Colouring(object):
    '''
        Abstract class for generators of k-colourings. Concrete classes
        will need to implement the `colours` method.
    '''
    def __init__(self, seed, k, device, *args, **kwargs):
        assert k > 0
        self.k = k
        self.seed = seed
        self.device = device
        self.generator = torch.Generator(device)
        self.reset()

    def colours(self, n, batch): # added batch as additional arg for colouring
        raise NotImplementedError

    def reset(self):
        self.generator = self.generator.manual_seed(self.seed)


class IndexColouringCentrality(Colouring):
    # TODO: rewrite the above
    def __init__(self, seed, nodes_to_colour, device, *args, **kwargs):
        self.num_nodes_color = nodes_to_colour
        super(IndexColouringCentrality, self).__init__(seed, nodes_to_colour, device, *args, **kwargs)

    def colours(self, n, batch):
        src = torch.ones_like(batch.batch)
        graph_sizes = torch_scatter.scatter_sum(src, batch.batch) # calculate graph sizes each time
        c = []
        xi = 0
        for graph_size in graph_sizes:
            centrality = batch.community[xi:xi+graph_size]
            xi += graph_size
            indices = centrality[:self.num_nodes_color]
            coloring = torch.zeros(graph_size)
            coloring[indices] = 1
            #coloring = F.one_hot(indices, num_classes=int(graph_size)).T
            c.append(coloring)

        return torch.cat(c,0).reshape(-1)


class Painter(object):

        def __init__(self, seed, k, nodes_to_colour, device, colouring_scheme='uniform', encoding='onehot', num_samples=1, **kwargs):
            assert k is not None and k > 0
            self.k = k
            self.seed = seed
            self.device = device
            self.encoding = encoding
            self.num_samples = num_samples
            self.colouring_scheme = colouring_scheme
            self.nodes_to_colour = nodes_to_colour
            if colouring_scheme in ['gaussian']:
                self.mu = kwargs['mu']
                self.sigma = kwargs['sigma']
            self.init_colouring()

        def colour_data(self, data, colours):
            '''
                Auxiliary routine which augments data objects with tensorial representations of colours.
                Arguments
                        `data`: the PyG data object to augment with colour representations
                        `colours`: sampled colours
            '''
            assert colours.shape[0] == data.num_nodes * data.num_samples
            if self.encoding == 'onehot':
                assert colours.ndim == 1
                c = F.one_hot(colours, self.k).to(torch.get_default_dtype())
                data.c_samples = c.reshape(-1, self.k*self.num_samples)
            elif self.encoding == 'onehot_2d':
                assert colours.ndim == 1
                c = torch.zeros(colours.shape[0], self.k, self.k)
                enc = F.one_hot(colours, self.k).to(torch.get_default_dtype())
                c[tuple(range(len(colours))), colours] = enc
                data.c_samples = c.reshape(-1, self.k*self.num_samples, self.k)
            elif self.encoding == 'identity':
                c = colours.to(torch.get_default_dtype())
                data.c_samples = c.reshape(-1, self.k*self.num_samples)
            else:
                raise NotImplementedError
            return data

        def paint(self, batch):
            '''
                Auxiliary routine which assigns colours to batch objects in an efficient way.
            '''
            n = batch.num_nodes * self.num_samples
            colours = self.colouring.colours(n, batch)
            batch.x_repeated = batch.x.repeat((1,self.num_samples))
            if batch.edge_attr is not None:
                if batch.edge_attr.ndim == 1:
                    batch.edge_attr_repeated = batch.edge_attr.unsqueeze(1).repeat((1,self.num_samples))
                else:
                    batch.edge_attr_repeated = batch.edge_attr.repeat((1,self.num_samples))
            batch.num_samples = self.num_samples
            batch.num_colours = self.k
            coloured_batch = self.colour_data(batch, colours)
            return coloured_batch

        def init_colouring(self):
            if self.colouring_scheme in ['indexsubgraph', 'indexrandom']:
                self.colouring = IndexColouringCentrality(self.seed, self.num_samples, self.device)
              
            else:
                raise ValueError(f"Unsupported scheme '{self.colouring_scheme}'")

        def reset(self):
            self.colouring.reset()
