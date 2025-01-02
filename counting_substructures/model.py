import torch
import torch.nn.functional as F

from conv import (GINConv, ColourCatGINConv, SharedGINConv, ColourCatSharedGINConv, CodeqSharedGINConv, 
    CodeqSharedGraphConv, ZINCGINConv, ColourCatSharedZINCGINConv, SharedZINCGINConv, DSSGINConv, ColourCatDSSGINConv,
    ColourCatDSSZINCGINConv, DSSZINCGINConv, PPAGINConv, ColourCatSharedPPAGINConv, SharedPPAGINConv)
from nn import Reshaper, ColourReshaper, ColourCatter

def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

class DeepSets(torch.nn.Module):
    '''
        Invariant DeepSets model in the 'simple' form, i.e. y = mlp_2( ∑_i mlp_1( x_i ) ) .
        In the following, adhering to the original paper, mlp_1 = phi, mlp_2 = rho.
        Here we assume sets all have the same size and are arranged in an input tensor of the form
        [batch_size, num_samples, in_dim]. Accordingly, the set pooling operation takes place
        over dimension 1.
    '''
    def __init__(self, in_dim, emb_dim, aggregator='sum', multiplier=1, bn=True):
        super(DeepSets, self).__init__()
        self.aggregator = aggregator
        self.bn = bn
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(in_dim, multiplier * emb_dim),
            torch.nn.BatchNorm1d(multiplier * emb_dim) if bn else torch.nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Linear(multiplier * emb_dim, emb_dim))
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(in_dim, multiplier * emb_dim),
            torch.nn.BatchNorm1d(multiplier * emb_dim) if bn else torch.nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Linear(multiplier * emb_dim, emb_dim))
    
    def reset_parameters(self):
        reset(self.phi)
        reset(self.rho)

    def forward(self, x):
        if self.bn:
            bs, card, _ = x.shape
            x = x.reshape(bs*card, -1)
            h = self.phi(x)
            h = h.reshape(bs, card, -1)
        else:
            h = self.phi(x)
        if self.aggregator=='sum':
            h = torch.sum(h, 1)
        elif self.aggregator=='mean':
            h = torch.mean(h, 1)
        elif self.aggregator=='max':
            h = torch.max(h, 1).values
        else:
            raise NotImplementedError(f"Aggregator '{self.aggregator}' not currently supported.")
        return self.rho(h)

class SharedDeepSets(torch.nn.Module):

    def __init__(self, in_dim, emb_dim, num_samples, aggregator='sum', multiplier=1, bn=True):
        super(SharedDeepSets, self).__init__()
        self.in_resh = ColourReshaper(in_dim, num_samples, out=False)
        self.ds = DeepSets(in_dim, emb_dim, aggregator=aggregator, multiplier=multiplier, bn=bn)
        self.out_resh = Reshaper(emb_dim, num_samples, out=True, stacked_samples=True)
    
    def reset_parameters(self):
        self.in_resh.reset_parameters()
        self.ds.reset_parameters()
        self.out_resh.reset_parameters()

    def forward(self, x):
        x = self.in_resh(x)
        h = self.ds(x)
        return self.out_resh(h)


class GNN(torch.nn.Module):
    '''
        Implements a standard Graph Neural Network for graph-wise tasks.
    '''

    def __init__(self, num_layers, in_dim, emb_dim, readout_module, prediction_module, layer='gin', residual=False, multiplier=1, bn=True, bn_between_convs=False):
        '''
            Arguments
                `num_layers`: number of stacked message-passing layers
                `in_dim`: input dimension
                `emb_dim`: dimension of hidden representations
                `readout_module`: module which maps node-wise embeddings into graph-wise ones; see "readout.py"
                `prediction_module`: module to output final model predictions, e.g. a classifier or a regressor
                `layer`: type of message-passing layer
                `residual`: whether to implement residual connections; note that, currently, they are supported
                    only from the second layer onwards
                `multiplier`: used only by convolutional layers which internally make use of MLPs; dictates how
                    much the dimension of hidden layers grow w.r.t. the input one
        '''
        super(GNN, self).__init__()
        assert num_layers > 0 and in_dim > 0 and emb_dim > 0
        self.num_layers = num_layers
        self.layer = layer
        self.convs = torch.nn.ModuleList()
        if bn_between_convs:
            self.batch_norms = torch.nn.ModuleList()
        else:
            self.batch_norms = None
        self.readout_module = readout_module
        self.prediction_module = prediction_module
        self.residual = residual
        self.bn = bn
        for l in range(num_layers):
            if layer == 'gin':
                self.convs.append(GINConv(emb_dim if l != 0 else in_dim, emb_dim, multiplier=multiplier, bn=bn))
            elif layer == 'zinc_gin':
                if l == 0:
                    self.convs.append(ZINCGINConv(in_dim, emb_dim, multiplier=multiplier, bn=bn, encode_input=True))
                else:
                    self.convs.append(ZINCGINConv(emb_dim, emb_dim, multiplier=multiplier, bn=bn, encode_input=False))
                  
            elif layer == 'ppa_gin':
                if l == 0:
                    self.convs.append(PPAGINConv(in_dim, emb_dim, multiplier=multiplier, bn=bn))
                else:
                    self.convs.append(PPAGINConv(emb_dim, emb_dim, multiplier=multiplier, bn=bn))
            else:
                raise ValueError("Unsupported layer '{}'".format(layer))
            if bn_between_convs:
                self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def reset_parameters(self):
        self.readout_module.reset_parameters()
        self.prediction_module.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()

    def forward(self, batched_data):
        '''
        Arguments
            `batched_data`: a batch of graphs arranged as a large(r) disconnected graph
        '''
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        current = x
        h_list = [current]
        for l in range(self.num_layers):
            h = self.convs[l](current, edge_index, edge_attr)
            if self.batch_norms is not None:
                h = self.batch_norms[l](h)
            if l != self.num_layers - 1:  # TODO: add conditions for more intrinsically non-linear conv layers
                h = F.relu(h)
            if l > 0 and self.residual:  # TODO: consider supporting residual connections at the first layer as well
                current = h + current
            else:
                current = h
            h_list.append(current)
        h = self.readout_module(h_list, batch)
        y = self.prediction_module(h)
        return y


class ColourCatGNN(torch.nn.Module):
    '''
        Implements a Coloured Graph Neural Network for graph-wise tasks.
        This model does not account for the possibility of colouring each graph in many different ways; it is the
        responsibility of end-users to implement and manage the averaging logic.
    '''

    def __init__(self, num_layers, in_dim, emb_dim, colour_dim, readout_module, prediction_module, layer='colourcat_gin',
        residual=False, multiplier=1, inject_colours=True, bn=True, bn_between_convs=False):
        '''
            Arguments
                `num_layers`: number of stacked message-passing layers
                `in_dim`: input dimension
                `emb_dim`: dimension of hidden representations
                `readout_module`: module which maps node-wise embeddings into graph-wise ones; see "readout.py"
                `prediction_module`: module to output final model predictions, e.g. a classifier or a regressor
                `layer`: type of message-passing layer
                `residual`: whether to implement residual connections; note that, currently, they are supported
                    only from the second layer onwards
                `multiplier`: used only by convolutional layers which internally make use of MLPs; dictates how
                    much the dimension of hidden layers grow w.r.t. the input one
                `inject_colours`: whether to inject the initial colours at every layer instead of simply using them at the first
                    layer only
        '''
        super(ColourCatGNN, self).__init__()
        self.num_layers = num_layers
        self.layer = layer
        self.convs = torch.nn.ModuleList()
        self.batch_norms = None
        if bn_between_convs:
            self.batch_norms = torch.nn.ModuleList()
        self.readout_module = readout_module
        self.prediction_module = prediction_module
        self.residual = residual
        self.inject_colours = inject_colours
        self.bn = bn
        for l in range(num_layers):
            if l > 0 and not inject_colours:
                layer = self.layer[len('colourcat_'):]
            else:
                layer = self.layer
            if layer == 'gin':
                self.convs.append(GINConv(emb_dim if l != 0 else in_dim, emb_dim, multiplier=multiplier, bn=bn))
            elif layer == 'colourcat_gin':
                self.convs.append(ColourCatGINConv(emb_dim if l != 0 else in_dim, emb_dim, colour_dim, multiplier=multiplier, bn=bn))
            else:
                raise ValueError("Unsupported layer '{}'".format(layer))
            if bn_between_convs:
                self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def reset_parameters(self):
        self.readout_module.reset_parameters()
        self.prediction_module.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()

    def forward(self, batched_data):
        x, edge_index, c, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.c, batched_data.edge_attr, \
                batched_data.batch
        current = x
        h_list = [current]
        for l in range(self.num_layers):
            if l > 0 and not self.inject_colours:
                h = self.convs[l](current, edge_index, edge_attr)
            else:
                h = self.convs[l](current, edge_index, c, edge_attr)
            if self.batch_norms is not None:
                h = self.batch_norms[l](h)
            if l != self.num_layers - 1:  # TODO: add conditions for more intrinsically non-linear conv layers
                h = F.relu(h)
            if l > 0 and self.residual:  # TODO: consider supporting residual connections at the first layer as well
                current = h + current
            else:
                current = h
            h_list.append(current)
        hs = self.readout_module(h_list, batch)
        y = self.prediction_module(hs)
        return y

class ColourCatSharedGNN(torch.nn.Module):
    '''
        Implements a Colour Catting Graph Neural Network for graph-wise tasks.
        Handles multiples samples for each input graph as well.
    '''

    def __init__(self, num_layers, in_dim, emb_dim, colour_dim, num_samples, readout_module, prediction_module, averaging_module, layer='colourcat_shared_gin',
        residual=False, multiplier=1, inject_colours=True, bn=True, bn_between_convs=False, sample_aggregation='none'):
        '''
            Arguments
                `num_layers`: number of stacked message-passing layers
                `in_dim`: input dimension
                `emb_dim`: dimension of hidden representations
                `readout_module`: module which maps node-wise embeddings into graph-wise ones; see "readout.py"
                `prediction_module`: module to output final model predictions, e.g. a classifier or a regressor
                `averaging_module`: module which aggregates final predictions for each graph according to different colourings
                `layer`: type of message-passing layer
                `residual`: whether to implement residual connections; note that, currently, they are supported
                    only from the second layer onwards
                `multiplier`: used only by convolutional layers which internally make use of MLPs; dictates how
                    much the dimension of hidden layers grow w.r.t. the input one
                `inject_colours`: whether to inject the initial colours at every layer instead of simply using them at the first
                    layer only
        '''
        super(ColourCatSharedGNN, self).__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.layer = layer
        self.convs = torch.nn.ModuleList()
        self.readout_module = readout_module
        self.prediction_module = prediction_module
        self.averaging_module = averaging_module
        self.residual = residual
        self.inject_colours = inject_colours
        self.bn = bn
        if bn_between_convs:
            self.in_resh = Reshaper(emb_dim, num_samples, out=False, stacked_samples=True)
            self.out_resh = Reshaper(emb_dim, num_samples, out=True, stacked_samples=True)
            self.batch_norms = torch.nn.ModuleList()
        else:
            self.batch_norms = None
        for l in range(num_layers):
            if l > 0 and not inject_colours:
                layer = self.layer[len('colourcat_'):]
            else:
                layer = self.layer
            if layer == 'shared_gin':
                self.convs.append(SharedGINConv(emb_dim if l != 0 else in_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn))
            elif layer == 'dss_gin':
                self.convs.append(DSSGINConv(emb_dim if l != 0 else in_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn, aggregation=sample_aggregation))
            elif layer == 'colourcat_shared_gin':
                self.convs.append(ColourCatSharedGINConv(emb_dim if l != 0 else in_dim, emb_dim, colour_dim, num_samples, multiplier=multiplier, bn=bn))
            elif layer == 'colourcat_dss_gin':
                self.convs.append(ColourCatDSSGINConv(emb_dim if l != 0 else in_dim, emb_dim, colour_dim, num_samples, multiplier=multiplier, bn=bn, aggregation=sample_aggregation))
            elif layer == 'colourcat_shared_zinc_gin':
                if l == 0:
                    self.convs.append(ColourCatSharedZINCGINConv(in_dim, emb_dim, colour_dim, num_samples, multiplier=multiplier, bn=bn, encode_input=True))
                else:
                    self.convs.append(ColourCatSharedZINCGINConv(emb_dim, emb_dim, colour_dim, num_samples, multiplier=multiplier, bn=bn, encode_input=False))
                    
            elif layer == 'colourcat_shared_ppa_gin':
                if l == 0:
                    self.convs.append(ColourCatSharedPPAGINConv(in_dim, emb_dim, colour_dim, num_samples, multiplier=multiplier, bn=bn, encode_input=True))
                else:
                    self.convs.append(ColourCatSharedPPAGINConv(emb_dim, emb_dim, colour_dim, num_samples, multiplier=multiplier, bn=bn, encode_input=False))
                    
            elif layer == 'shared_zinc_gin':
                if l == 0:
                    self.convs.append(SharedZINCGINConv(in_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn, encode_input=True))
                else:
                    self.convs.append(SharedZINCGINConv(emb_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn, encode_input=False))
                    
            elif layer == 'shared_ppa_gin':
                if l == 0:
                    self.convs.append(SharedPPAGINConv(in_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn))
                else:
                    self.convs.append(SharedPPAGINConv(emb_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn))
            elif layer == 'colourcat_dss_zinc_gin':
                if l == 0:
                    self.convs.append(ColourCatDSSZINCGINConv(in_dim, emb_dim, colour_dim, num_samples, multiplier=multiplier, bn=bn, encode_input=True))
                else:
                    self.convs.append(ColourCatDSSZINCGINConv(emb_dim, emb_dim, colour_dim, num_samples, multiplier=multiplier, bn=bn, encode_input=False))
            elif layer == 'dss_zinc_gin':
                if l == 0:
                    self.convs.append(DSSZINCGINConv(in_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn, encode_input=True))
                else:
                    self.convs.append(DSSZINCGINConv(emb_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn, encode_input=False))
            else:
                raise ValueError("Unsupported layer '{}'".format(layer))
            if bn_between_convs:
                self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def reset_parameters(self):
        self.readout_module.reset_parameters()
        self.prediction_module.reset_parameters()
        self.averaging_module.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()

    def forward(self, batched_data):
        x_repeated = batched_data.x_repeated
        c_samples = batched_data.c_samples
        edge_index = batched_data.edge_index
        edge_attr_repeated = None
        if batched_data.edge_attr is not None:
            edge_attr_repeated = batched_data.edge_attr_repeated
        batch = batched_data.batch
        try:
            sample_index = batched_data.sample_index
            sample_batch = batched_data.sample_batch
        except AttributeError:
            sample_index = None
            sample_batch = None
        current = x_repeated
        h_list = [current]
        for l in range(self.num_layers):
            if l > 0 and not self.inject_colours:
                h = self.convs[l](current, edge_index, edge_attr_repeated)
            else:
                h = self.convs[l](current, edge_index, c_samples, edge_attr_repeated)
            if self.batch_norms is not None:
                h = self.in_resh(h)
                h = self.batch_norms[l](h)
                h = self.out_resh(h)
            if l != self.num_layers - 1:  # TODO: add conditions for more intrinsically non-linear conv layers
                h = F.relu(h)
            if l > 0 and self.residual:  # TODO: consider supporting residual connections at the first layer as well
                current = h + current
            else:
                current = h
            h_list.append(current)
        #print(h_list[-1].shape)
        #print(h_list[-1][batch==0].shape)
        #print(h_list[-1][batch==0])
        #print(c_samples.shape)
        ### ext_c_samples = c_samples.repeat_interleave(self.emb_dim, dim=1)
        ### mask = ext_c_samples == 0
        #print(mask.shape)
        #print(h_list[-1].shape)
        #print(mask[0])
        #print(h_list[-1][mask])
        ### x = h_list[-1]
        ### x[mask] = 0
        ### h_list[-1] = x
        #print(h_list[-1][0])
        hs = self.readout_module(h_list, batch)
        #print(hs.shape)
        ys = self.prediction_module(hs)
        #print(ys.shape)
        if sample_index is not None and sample_batch is not None:
            y = self.averaging_module(ys, sample_index, sample_batch)
        else:
            y = self.averaging_module(ys)
        #print(y.shape)
        return y

class CodeqSharedGNN(torch.nn.Module):
    '''
        Implements a Code Equivariant Graph Neural Network for graph-wise tasks.
        Handles multiples samples for each input graph as well.
    '''
    # TODO: implement batch norm between layers

    def __init__(self, num_layers, in_dim, emb_dim, colour_in_dim, colour_emb_dim, num_colours, num_samples, readout_module, colour_readout_module, prediction_module, averaging_module,
        layer='codeq_shared_gin', residual=False, colour_picker_agg='sum', multiplier=1, bn=True, picker_b4_readout=False):
        '''
            Arguments
                `num_layers`: number of stacked message-passing layers
                `in_dim`: input dimension
                `emb_dim`: dimension of hidden representations
                `colour_emb_dim`: 
                `num_colours`: 
                `readout_module`: module which maps node-wise embeddings into graph-wise ones; see "readout.py"
                `colour_readout_module`:
                `prediction_module`: module to output final model predictions, e.g. a classifier or a regressor
                `averaging_module`: module which aggregates final predictions for each graph according to different colourings
                `layer`: type of message-passing layer
                `residual`: whether to implement residual connections; note that, currently, they are supported
                    only from the second layer onwards
                `colour_picker_agg`: the aggregator used in the colour picker layer (a Deep Sets)
                `multiplier`: used only by convolutional layers which internally make use of MLPs; dictates how
                    much the dimension of hidden layers grow w.r.t. the input one
        '''
        # TODO: add support for batch norm in between convolutional layers
        super(CodeqSharedGNN, self).__init__()
        self.num_layers = num_layers
        self.layer = layer
        self.convs = torch.nn.ModuleList()
        self.readout_module = readout_module
        self.colour_readout_module = colour_readout_module
        self.prediction_module = prediction_module
        self.averaging_module = averaging_module
        self.residual = residual
        self.picker_b4_readout = picker_b4_readout
        for l in range(num_layers):
            if layer == 'codeq_shared_gin':
                self.convs.append(CodeqSharedGINConv(emb_dim if l != 0 else in_dim, emb_dim, colour_emb_dim if l!= 0 else colour_in_dim, colour_emb_dim, 
                    num_samples, multiplier=multiplier, bn=bn))
            elif layer == 'codeq_shared_graphconv':
                self.convs.append(CodeqSharedGraphConv(emb_dim if l != 0 else in_dim, emb_dim, colour_emb_dim if l!= 0 else colour_in_dim, colour_emb_dim,
                    num_samples))
            else:
                raise ValueError("Unsupported layer '{}'".format(layer))
        self.colour_picker = SharedDeepSets(colour_emb_dim, colour_emb_dim, num_samples, aggregator=colour_picker_agg, multiplier=multiplier, bn=bn)
        self.colour_catter = ColourCatter(emb_dim, colour_emb_dim, num_samples, out_reshape=True)

    def reset_parameters(self):
        self.readout_module.reset_parameters()
        self.prediction_module.reset_parameters()
        self.averaging_module.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.colour_readout.reset_parameters()

    def forward(self, batched_data):

        x_repeated = batched_data.x_repeated
        c_samples = batched_data.c_samples
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr  # TODO: implement handling of edge attributes repeated for the various samples
        batch = batched_data.batch
        current_x = x_repeated
        current_c = c_samples
        h_x_list = [current_x]
        h_c_list = [current_c]
        for l in range(self.num_layers):
            h_x, h_c = self.convs[l](current_x, current_c, edge_index, edge_attr)
            if l != self.num_layers - 1:  # TODO: add conditions for more intrinsically non-linear conv layers
                h_x = F.relu(h_x)
                h_c = F.relu(h_c)
            if l > 0 and self.residual:  # TODO: consider supporting residual connections at the first layer as well
                current_x = h_x + current_x
                current_c = h_c + current_c
            else:
                current_x = h_x
                current_c = h_c
            h_x_list.append(current_x)
            h_c_list.append(current_c)
        h_xs = self.readout_module(h_x_list, batch)
        if self.picker_b4_readout:
            # here we test a possible approach where colours are summarised into just one vector
            # before being read out over all nodes like in the case of standard features
            # TODO: (!) test / no jk!
            h_cs = self.colour_picker(h_c_list[-1])
            h_cs = self.readout_module([h_cs], batch)
        else:
            # here the alternative branch: we first readout colours and then summarise them into one vector
            h_cs = self.colour_readout_module(h_c_list, batch)
            h_cs = self.colour_picker(h_cs)
        hs = self.colour_catter(h_xs, h_cs)
        ys = self.prediction_module(hs)
        y = self.averaging_module(ys)
        return y

'''
    When adding a new coloured model, remember to properly list it in `constants.py`!
'''
