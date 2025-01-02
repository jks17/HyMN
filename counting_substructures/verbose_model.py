import torch
import torch.nn.functional as F

from conv import ColourCatGINConv, GINConv

class VerboseColourCatGNN(torch.nn.Module):
    '''
        Implements a Coloured Graph Neural Network for graph-wise tasks.
        Essentially, each input graph is received multiple times, with nodes coloured differently each time.
        The model may consider these colours to pilot the message-passing operations, than the representations
        (predictions) from all colourings of the same graph are aggregated via a generalised `averaging` operation.
    '''
    # TODO: implement batch norm between layers
    def __init__(self, num_layers, in_dim, emb_dim, colour_dim, readout_module, prediction_module, averaging_module, layer='colourcat_gin',
        residual=False, multiplier=1, inject_colours=True, bn=True):
        '''
            Arguments
                `num_layers`: number of stacked message-passing layers
                `in_dim`: input dimension
                `emb_dim`: dimension of hidden representations
                `readout_module`: module which maps node-wise embeddings into graph-wise ones; see "readout.py"
                `prediction_module`: module to output final model predictions, e.g. a classifier or a regressor
                `averaging_module`: module which maps coloured graph-wise embeddings into a single graph-wise one; see "readout.py"
                `layer`: type of message-passing layer
                `residual`: whether to implement residual connections; note that, currently, they are supported
                    only from the second layer onwards
                `multiplier`: used only by convolutional layers which internally make use of MLPs; dictates how
                    much the dimension of hidden layers grow w.r.t. the input one
                `inject_colours`: whether to inject the initial colours at every layer instead of simply using them at the first
                    layer only
        '''
        super(VerboseColourCatGNN, self).__init__()
        self.num_layers = num_layers
        self.layer = layer
        self.convs = torch.nn.ModuleList()
        self.readout_module = readout_module
        self.prediction_module = prediction_module
        self.averaging_module = averaging_module
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

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, batched_data):
        x, edge_index, c, edge_attr, batch, colouring2graph = batched_data.x, batched_data.edge_index, batched_data.c, batched_data.edge_attr, \
            batched_data.batch, batched_data.colouring2graph
        current = x
        h_list = [current]
        for l in range(self.num_layers):
            if l > 0 and not self.inject_colours:
                h = self.convs[l](current, edge_index, edge_attr)
            else:
                h = self.convs[l](current, edge_index, c, edge_attr)
            if l != self.num_layers - 1:  # TODO: add conditions for more intrinsically non-linear conv layers
                h = F.relu(h)
            if l > 0 and self.residual:  # TODO: consider supporting residual connections at the first layer as well
                current = h + current
            else:
                current = h
            h_list.append(current)
        hs = self.readout_module(h_list, batch)
        ys = self.prediction_module(hs)
        y = self.averaging_module(ys, colouring2graph)  # TODO: consider averaging before the predictions?
        return y
