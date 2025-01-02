import torch


from torch_geometric.nn import GINConv as PyGINConv, GINEConv as PyGINEConv
from torch_geometric.nn import MessagePassing

from nn import (reset, Reshaper, ColourCatter, SharedLinear, SharedMLP, 
    ColourCatSharedLinear, ColourCatSharedMLP, CodeqSharedMLP, CodeqSharedLinear)

def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

class GINConv(torch.nn.Module):

    def __init__(self, in_dim, emb_dim, multiplier=1, bn=True):
        super(GINConv, self).__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, multiplier * emb_dim),
            torch.nn.BatchNorm1d(multiplier * emb_dim) if bn else torch.nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Linear(multiplier * emb_dim, emb_dim)
        )
        self.layer = PyGINConv(nn=mlp, train_eps=True)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        return self.layer(x, edge_index)

class SharedGINConv(torch.nn.Module):
    '''
        GIN convolutional layer which runs on multiple feature-vectors per node at the same time.
    '''
    def __init__(self, in_dim, emb_dim, num_samples, multiplier=1, bn=True):
        super(SharedGINConv, self).__init__()
        mlp = SharedMLP(in_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn)
        self.layer = PyGINConv(nn=mlp, train_eps=True)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        return self.layer(x, edge_index)

class DSSGINConv(torch.nn.Module):

    def __init__(self, in_dim, emb_dim, num_samples, multiplier=1, bn=True, aggregation='mean'):
        super(DSSGINConv, self).__init__()
        self.shared_layer = SharedGINConv(in_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn)
        self.layer = GINConv(in_dim, emb_dim, multiplier=multiplier, bn=bn)
        self.in_resh = Reshaper(in_dim, num_samples, out=False, stacked_samples=False)
        self.mid_resh = Reshaper(emb_dim, num_samples, out=False, stacked_samples=False)
        self.out_resh = Reshaper(emb_dim, num_samples, out=True, stacked_samples=False)
        assert aggregation in ['sum', 'mean']
        self.aggregation = aggregation

    def reset_parameters(self):
        self.shared_layer.reset_parameters()
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        if self.aggregation == 'sum':
            aggregated_in = torch.sum(self.in_resh(x), 1)
        else:
            aggregated_in = torch.mean(self.in_resh(x), 1)
        siamese_out = self.mid_resh(self.shared_layer(x, edge_index, edge_attr))
        aggregated_out = self.layer(aggregated_in, edge_index, edge_attr)
        fedback = siamese_out + torch.unsqueeze(aggregated_out, 1)
        return self.out_resh(fedback)
    


class ColourCatGINConv(torch.nn.Module):

    def __init__(self, in_dim, emb_dim, colour_dim, multiplier=1, bn=True):
        super(ColourCatGINConv, self).__init__()
        self.colour_catter = ColourCatter(in_dim, colour_dim, 1, out_reshape=True)
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim+colour_dim, multiplier * emb_dim),
            torch.nn.BatchNorm1d(multiplier * emb_dim) if bn else torch.nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Linear(multiplier * emb_dim, emb_dim)
        )
        self.layer = PyGINConv(nn=mlp, train_eps=True)

    def reset_parameters(self):
        self.colour_catter.reset_parameters()
        self.layer.reset_parameters()

    def forward(self, x, edge_index, c, edge_attr=None):
        assert x.shape[0] == c.shape[0]
        h = self.colour_catter(x, c)
        return self.layer(h, edge_index)

class ColourCatSharedGINConv(torch.nn.Module):
    '''
        GIN convolutional layer which runs on the concatenation of colours and features and on
        multiple colouring samples at the same time.
    '''
    def __init__(self, in_dim, emb_dim, colour_dim, num_samples, multiplier=1, bn=True):
        super(ColourCatSharedGINConv, self).__init__()
        self.colour_catter = ColourCatter(in_dim, colour_dim, num_samples, out_reshape=True)
        mlp = SharedMLP(in_dim+colour_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn)
        self.layer = PyGINConv(nn=mlp, train_eps=True)

    def reset_parameters(self):
        self.colour_catter.reset_parameters()
        self.layer.reset_parameters()

    def forward(self, x, edge_index, c, edge_attr=None):
        assert x.shape[0] == c.shape[0]
        h = self.colour_catter(x, c)
        return self.layer(h, edge_index)

class ColourCatDSSGINConv(torch.nn.Module):

    def __init__(self, in_dim, emb_dim, colour_dim, num_samples, multiplier=1, bn=True, aggregation='mean'):
        super(ColourCatDSSGINConv, self).__init__()
        self.shared_layer = ColourCatSharedGINConv(in_dim, emb_dim, colour_dim, num_samples, multiplier=multiplier, bn=bn)
        self.layer = GINConv(in_dim+colour_dim, emb_dim, multiplier=multiplier, bn=bn)
        self.colour_catter = ColourCatter(in_dim, colour_dim, num_samples, out_reshape=False)
        self.mid_resh = Reshaper(emb_dim, num_samples, out=False, stacked_samples=False)
        self.out_resh = Reshaper(emb_dim, num_samples, out=True, stacked_samples=False)
        assert aggregation in ['sum', 'mean']
        self.aggregation = aggregation

    def reset_parameters(self):
        self.shared_layer.reset_parameters()
        self.layer.reset_parameters()

    def forward(self, x, edge_index, c, edge_attr=None):
        if self.aggregation == 'sum':
            aggregated_in = torch.sum(self.colour_catter(x, c), 1)
        else:
            aggregated_in = torch.mean(self.colour_catter(x, c), 1)
        siamese_out = self.mid_resh(self.shared_layer(x, edge_index, c, edge_attr=edge_attr))
        aggregated_out = self.layer(aggregated_in, edge_index, edge_attr=edge_attr)
        fedback = siamese_out + torch.unsqueeze(aggregated_out, 1)
        return self.out_resh(fedback)
    

class CodeqMessagePassing(MessagePassing):

    def message_feats(self, x_j, c_j):
        return x_j

    def message_colours(self, x_j, c_j):
        return c_j

    def message(self, x_j, c_j, mode):
        if mode == 'feats':
            return self.message_feats(x_j, c_j)
        elif mode == 'colours':
            return self.message_colours(x_j, c_j)
        else:
            raise ValueError(f"Unknown mode '{mode}'")

    def update_feats(self, inp, x_i, x_j, c_i, c_j):
        return inp

    def update_colours(self, inp, x_i, x_j, c_i, c_j):
        return inp

    def update(self, inp, x_i, x_j, c_i, c_j, mode):
        if mode == 'feats':
            return self.update_feats(inp, x_i, x_j, c_i, c_j)
        elif mode == 'colours':
            return self.update_colours(inp, x_i, x_j, c_i, c_j)
        else:
            raise ValueError(f"Unknown mode '{mode}'")

    def _c_intro(self, c):
        assert c.ndim in [2, 3]
        if c.ndim == 2:
            c = c.unsqueeze(-1)
        c = c.flatten(start_dim=-2, end_dim=-1)
        return c

    def _c_outro(self, out_c, dim):
        out_c = out_c.unflatten(-1, (-1, dim))
        return out_c

class CodeqSharedGINConv(CodeqMessagePassing):

    def __init__(self, in_dim, emb_dim, colour_in_dim, colour_emb_dim, num_samples, multiplier=1, bn=True):
        super(CodeqSharedGINConv, self).__init__()
        self.nn = CodeqSharedMLP(in_dim, colour_in_dim, emb_dim, colour_emb_dim, num_samples, multiplier=multiplier, bn=bn)
        self.eps_x = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.eps_c = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.colour_in_dim = colour_in_dim
        self.reset_parameters()

    def reset_parameters(self):
        self.nn.reset_parameters()
        self.eps_x.data.fill_(0.0)
        self.eps_c.data.fill_(0.0)

    def forward(self, x, c, edge_index, edge_attr=None):

        '''
            x: [num_nodes, num_samples*in_dim]
            c: [num_nodes, num_samples*num_colours, colour_in_dim]
                   ->
            c: [num_nodes, num_samples*num_colours*colour_in_dim]
                   ->
            out_x: [num_nodes, num_samples*in_dim]
            out_c: [num_nodes, num_samples*num_colours*colour_in_dim]
                   ->
            out_c: [num_nodes, num_samples*num_colours, colour_in_dim]
                   ->
            y_x: [num_nodes, num_samples*emb_dim]
            y_c: [num_nodes, num_samples*num_colours, colour_emb_dim]
        '''
        c = self._c_intro(c)
        out_x = self.propagate(edge_index, x=x, c=c, mode='feats')
        out_c = self.propagate(edge_index, x=x, c=c, mode='colours')
        out_x = out_x + (1.0 + self.eps_x) * x
        out_c = out_c + (1.0 + self.eps_c) * c
        out_c = self._c_outro(out_c, self.colour_in_dim)
        return self.nn(out_x, out_c)

    def __repr__(self):
        return f'{self.__class__.__name__}(nn={self.nn})'

class CodeqSharedGraphConv(CodeqMessagePassing):

    def __init__(self, in_dim, emb_dim, colour_in_dim, colour_emb_dim, num_samples):
        super(CodeqSharedGraphConv, self).__init__()
        self.linear_self = CodeqSharedLinear(in_dim, colour_in_dim, emb_dim, colour_emb_dim, num_samples)
        self.linear_msg = CodeqSharedLinear(in_dim, colour_in_dim, emb_dim, colour_emb_dim, num_samples)
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.colour_in_dim = colour_in_dim
        self.colour_emb_dim = colour_emb_dim
        self.reset_parameters()

    def reset_parameters(self):
        self.linear_self.reset_parameters()
        self.linear_msg.reset_parameters()

    def forward(self, x, c, edge_index, edge_attr=None):

        '''
            x:           [num_nodes, num_samples*in_dim]
            c:           [num_nodes, num_samples*num_colours, colour_in_dim]
            c_flat:      [num_nodes, num_samples*num_colours*colour_in_dim]
            out_x:       [num_nodes, num_samples*emb_dim]
            out_c_flat:  [num_nodes, num_samples*num_colours*colour_emb_dim]
            out_c:       [num_nodes, num_samples*num_colours, colour_emb_dim]

          x ----------------------------+(propagate)---------------------> out_x
          |                            /                                     |
          |  c --(flatten)--> c_flat -+----+(propagate)--> out_c_flat        |
          |  |                            /                    |             |
          +-_|_--------------------------+                (unflatten)        |
          |  |                                                 |             |
          |  |                                                 v             |
          |  |                                               out_c           |
          |  |                                                 |             |
          |  |                                                 v             |
          |  +--+(linear_self)+------------------------------>(+)-->>>       |
          |    /               \\                                            v
          +---+                 +------------------------------------------>(+)-->>>

        '''
        c_flat = self._c_intro(c)
        out_x = self.propagate(edge_index, x=x, c=c_flat, mode='feats')
        out_c_flat = self.propagate(edge_index, x=x, c=c_flat, mode='colours')
        out_c = self._c_outro(out_c_flat, self.colour_emb_dim)
        out_x = out_x + self.linear_self.x_fwd(x, c)
        out_c = out_c + self.linear_self.c_fwd(x, c)
        return out_x, out_c

    def message_feats(self, x_j, c_j):
        '''
            x_j ---------------------------+(linear_msg, x_fwd)-->>>
                                          /
            c_j --(unflatten)--> c_j_unf +
        '''
        c_j_unf = c_j.unflatten(-1, (-1, self.colour_in_dim))
        return self.linear_msg.x_fwd(x_j, c_j_unf)

    def message_colours(self, x_j, c_j):
        '''
            x_j --------------------------+
                                           \
            c_j --(unflatten)--> c_j_unf ---+(linear_msg, c_fwd)--(flatten)-->>>
        '''
        c_j_unf = c_j.unflatten(-1, (-1, self.colour_in_dim))
        y = self.linear_msg.c_fwd(x_j, c_j_unf)
        return y.flatten(start_dim=-2, end_dim=-1)

    def __repr__(self):
        return f'{self.__class__.__name__}(lin_self={self.linear_self}, lin_msg={self.linear_msg})'
