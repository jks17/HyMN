import torch

from torch.nn import functional as F

def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

class Reshaper(torch.nn.Module):
    '''
        Auxiliary module to flatten, unflatten or simply reshape inputs.
    '''
    def __init__(self, num_feats, num_samples, out=False, stacked_samples=True):
        super(Reshaper, self).__init__()
        self.out = out
        self.num_feats = num_feats
        self.num_samples = num_samples
        self.stacked_samples = stacked_samples

    def reset_parameters(self):
        return

    def forward(self, x):
        if self.stacked_samples:
            # 2d to 2d, but just in a way that the last dimension only contains feats
            if self.out:
                assert (x.shape[0] % self.num_samples) == 0
                num_nodes = int(x.shape[0] / self.num_samples)
                return x.reshape(num_nodes, self.num_samples*self.num_feats)
            else:
                '''
                    [[feat_11, feat_12],
                     [feat_21, feat_22]]
                =>
                    [[feat_11],
                     [feat_12],
                     [feat_21],
                     [feat_22]]
                '''
                num_nodes = x.shape[0]
                return x.reshape(num_nodes*self.num_samples, self.num_feats)
        else:
            # 2d to 3d (and viceversa), in a way that samples are gathered
            # separately than features
            if self.out:
                return x.reshape(-1, self.num_samples*self.num_feats)
            else:
                '''
                    [[feat_11, feat_12],
                     [feat_21, feat_22]]
                =>
                    [[[feat_11],
                      [feat_21]],
                     [[feat_12],
                      [feat_22]]]
                '''
                return x.reshape(-1, self.num_samples, self.num_feats)

class ColourReshaper(torch.nn.Module):
    '''
        Auxiliary module to flatten, unflatten or simply reshape input colours.
        For now, we only support 3d <-> 3d. Here, 3d referes to the tensors in the form
        [num_nodes, num_colours*num_samples, num_feats] and the reshaped tensor
        [num_nodes*num_samples, num_colours,num_feats]. Apparently, 3d <-> 4d, where 4d
        refers to the shape of tensor [num_nodes, num_samples, num_colours, num_feats],
        is not trivial to implement. In any case, it is not clear whether this last
        approach is needed at all: for BN we would like to adopt the 3d <-> 3d strategy. 
    '''
    def __init__(self, num_feats, num_samples, out=False):
        super(ColourReshaper, self).__init__()
        self.out = out
        self.num_feats = num_feats
        self.num_samples = num_samples

    def reset_parameters(self):
        return

    def forward(self, c):
        # 3d to 3d, but in a way that samples are arranged over the first dimension.
        if self.out:
            '''
                [num_nodes*num_samples, num_colours, num_feats]
                                        ->
                [num_nodes, num_colours*num_samples, num_feats]
            '''
            assert (c.shape[0] % self.num_samples) == 0
            num_nodes = int(c.shape[0] / self.num_samples)
            return c.reshape(num_nodes, -1, self.num_feats)
        else:
            '''
                [num_nodes, num_colours*num_samples, num_feats]
                                        ->
                [num_nodes*num_samples, num_colours, num_feats]

                E.g.:

                [[[colour_111_a, colour_111_b],
                  [colour_112_a, colour_112_b],
                  [colour_121_a, colour_121_b],
                  [colour_122_a, colour_122_b]],
                 [[colour_211_a, colour_211_b],
                  [colour_212_a, colour_212_b],
                  [colour_221_a, colour_221_b],
                  [colour_222_a, colour_222_b]]]
                               ->
                [[[colour_111_a, colour_111_b],
                  [colour_121_a, colour_121_b]],
                 [[colour_112_a, colour_112_b],
                  [colour_122_a, colour_122_b]],
                 [[colour_211_a, colour_211_b],
                  [colour_221_a, colour_221_b]],
                 [[colour_212_a, colour_212_b],
                  [colour_222_a, colour_222_b]]]
            '''
            assert (c.shape[1] % self.num_samples) == 0
            num_colours = int(c.shape[1] / self.num_samples)
            num_nodes = c.shape[0]
            return c.reshape(num_nodes*self.num_samples, num_colours, self.num_feats)

class ColourCatter(torch.nn.Module):
    '''
        Auxiliary module to concatenate colours to features and jointly
        properly flatten / unflatten if required.
    '''
    def __init__(self, num_feats, num_colours, num_samples, out_reshape=True):
        super(ColourCatter, self).__init__()
        self.feat_resh = Reshaper(num_feats, num_samples, out=False, stacked_samples=False)
        self.colour_resh = Reshaper(num_colours, num_samples, out=False, stacked_samples=False)
        self.out_resh = None
        if out_reshape:
            self.out_resh = Reshaper(num_feats+num_colours, num_samples, out=True, stacked_samples=False)

    def reset_parameters(self):
        self.feat_resh.reset_parameters()
        self.colour_resh.reset_parameters()
        self.out_resh.reset_parameters()

    def forward(self, x, c):
        x = self.feat_resh(x)
        c = self.colour_resh(c)
        h = torch.cat([x, c], dim=2)
        if self.out_resh is not None:
            h = self.out_resh(h)
        return h

class SharedLinear(torch.nn.Module):
    '''
        A Linear layer which operates on a flattened, but 3d, object.
    '''
    def __init__(self, in_dim, emb_dim, num_samples):
        super(SharedLinear, self).__init__()
        self.linear = torch.nn.Linear(in_dim, emb_dim)
        self.in_resh = Reshaper(in_dim, num_samples, out=False, stacked_samples=True)
        self.out_resh = Reshaper(emb_dim, num_samples, out=True, stacked_samples=True)
        self.layers = torch.nn.Sequential(
            self.in_resh,
            self.linear,
            self.out_resh)

    def reset_parameters(self):
        reset(self.layers)

    def forward(self, x):
        return self.layers(x)

class SharedMLP(torch.nn.Module):
    '''
        An MLP which operates on a flattened, but 3d, object.
    '''
    def __init__(self, in_dim, emb_dim, num_samples, multiplier=1, bn=True):
        super(SharedMLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, multiplier * emb_dim),
            torch.nn.BatchNorm1d(multiplier * emb_dim) if bn else torch.nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Linear(multiplier * emb_dim, emb_dim))
        self.in_resh = Reshaper(in_dim, num_samples, out=False, stacked_samples=True)
        self.out_resh = Reshaper(emb_dim, num_samples, out=True, stacked_samples=True)
        self.layers = torch.nn.Sequential(
            self.in_resh,
            self.mlp,
            self.out_resh)

    def reset_parameters(self):
        reset(self.layers)

    def forward(self, x):
        return self.layers(x)

class ColourCatSharedLinear(torch.nn.Module):
    '''
        Same as SharedLinear, but it runs on the concatenation of colours and features.
    '''
    def __init__(self, in_dim, emb_dim, colour_dim, num_samples):
        super(ColourCatSharedLinear, self).__init__()
        self.colour_catter = ColourCatter(in_dim, colour_dim, num_samples, out_reshape=True)
        self.linear = SharedLinear(in_dim+colour_dim, emb_dim, num_samples)

    def reset_parameters(self):
        self.colour_catter.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x):
        h = self.colour_catter(x, c)
        return self.linear(h)

class ColourCatSharedMLP(torch.nn.Module):
    '''
        Same as SharedMLP, but it runs on the concatenation of colours and features.
    '''
    def __init__(self, in_dim, emb_dim, colour_dim, num_samples, multiplier=1, bn=True):
        super(ColourCatSharedMLP, self).__init__()
        self.colour_catter = ColourCatter(in_dim, colour_dim, num_samples, out_reshape=True)
        self.mlp = SharedMLP(in_dim+colour_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn)

    def reset_parameters(self):
        self.colour_catter.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x, c):
        h = self.colour_catter(x, c)
        return self.mlp(h)

class DeepSetLinear(torch.nn.Module):
    '''
        A linear, equivariant set layer operating on a set as a tensor [num_elements, num_feats].
        It can work on a batch of sets as well, assuming they all have the same card.
        In this case, the batch is a tensor [num_sets, num_elements, num_feats].
    '''
    def __init__(self, in_dim, emb_dim):
        super(DeepSetLinear, self).__init__()
        self.lamda = torch.nn.Linear(in_dim, emb_dim)
        self.gamma = torch.nn.Linear(in_dim, emb_dim, bias=False)
        # TODO: check bias term implemented like this is just correct

    def reset_parameters(self):
        self.lamda.reset_parameters()
        self.gamma.reset_parameters()

    def forward(self, x):
        h = self.lamda(x)
        aggr = torch.sum(x, -2, keepdims=True)
        # NB: torch takes care of broadcasting automatically if we keep the full dimension for `aggr`
        h += self.gamma(aggr)
        return h

class Codeq1dLinear(torch.nn.Module):
    '''
        A linear layer which updates coloured inputs and is equivariant to permutations of colours.
        It is assumed that colours are always 1d, so that the input tensors x, c are in the form:
            x: [num_nodes (or batch_size), in_dim]
            c: [num_nodes (or batch_size), num_colours]
    '''
    # TODO: Properly handle bias terms...
    def __init__(self, in_dim, emb_dim):
        super(Codeq1dLinear, self).__init__()
        self.f2f = torch.nn.Linear(in_dim, emb_dim)
        self.c2f = torch.nn.Linear(1, emb_dim)
        self.f2c = torch.nn.Linear(in_dim, 1)
        self.c2c = DeepSetLinear(1, 1)

    def reset_parameters(self):
        self.f2f.reset_parameters()
        self.f2c.reset_parameters()
        self.c2f.reset_parameters()
        self.c2c.reset_parameters()

    def forward(self, x, c):
        x_ = self.f2f(x)
        x_ += self.c2f(torch.sum(c, 1, keepdims=True))
        c_ = torch.unsqueeze(c, 2)
        c_ = torch.squeeze(self.c2c(c_), -1)
        # NB: broadcasting is performed automatically here
        c_ += self.f2c(x)
        return x_, c_

class CodeqLinear(torch.nn.Module):
    '''
        A linear layer which updates coloured inputs and is equivariant to permutations of colours.
        It is assumed that colours can have any number of channels associated to them, so that the
        input tensors x, c are in the form:
            x: [num_nodes (or batch_size), in_dim]
            c: [num_nodes (or batch_size), num_colours, colour_in_dim]
    '''
    # TODO: Properly handle bias terms...
    def __init__(self, in_dim, colour_in_dim, emb_dim, colour_emb_dim):
        super(CodeqLinear, self).__init__()
        self.f2f = torch.nn.Linear(in_dim, emb_dim)
        self.c2f = torch.nn.Linear(colour_in_dim, emb_dim)
        self.f2c = torch.nn.Linear(in_dim, colour_emb_dim)
        self.c2c = DeepSetLinear(colour_in_dim, colour_emb_dim)

    def reset_parameters(self):
        self.f2f.reset_parameters()
        self.f2c.reset_parameters()
        self.c2f.reset_parameters()
        self.c2c.reset_parameters()

    def x_fwd(self, x, c):
        # [-1, in_dim] -> [-1, emb_dim]
        x_ = self.f2f(x)
        # [-1, num_colours, c_in_dim] -> [-1, c_in_dim] -> [-1, emb_dim]
        x_ += self.c2f(torch.sum(c, 1, keepdims=False))
        return x_

    def c_fwd(self, x, c):
        # [-1, num_colours, c_in_dim] -> [-1, num_colours, c_emb_dim]
        c_ = self.c2c(c)
        # [-1, in_dim] -> [-1, c_emb_dim] -> [-1, 1, c_emb_dim] -> [-1, num_colours, c_emb_dim]
        # NB: broadcasting is performed automatically here
        c_ += self.f2c(x).unsqueeze(1)
        return c_

    def forward(self, x, c):
        x_ = self.x_fwd(x, c)
        c_ = self.c_fwd(x, c)
        return x_, c_

class CodeqSharedLinear(torch.nn.Module):
    '''
        Pass
    '''
    # TODO: Properly handle bias terms...
    def __init__(self, in_dim, colour_in_dim, emb_dim, colour_emb_dim, num_samples):
        super(CodeqSharedLinear, self).__init__()
        self.x_in_resh = Reshaper(in_dim, num_samples, out=False, stacked_samples=True)
        self.x_out_resh = Reshaper(emb_dim, num_samples, out=True, stacked_samples=True)
        self.c_in_resh = ColourReshaper(colour_in_dim, num_samples, out=False)
        self.c_out_resh = ColourReshaper(colour_emb_dim, num_samples, out=True)
        self.codeq_linear = CodeqLinear(in_dim, colour_in_dim, emb_dim, colour_emb_dim)

    def reset_parameters(self):
        self.x_in_resh.reset_parameters()
        self.x_out_resh.reset_parameters()
        self.c_in_resh.reset_parameters()
        self.c_out_resh.reset_parameters()
        self.codeq_linear.reset_parameters()

    def x_fwd(self, x, c):
        x_ = self.x_in_resh(x)
        c_ = self.c_in_resh(c)
        x_ = self.codeq_linear.x_fwd(x_, c_)
        x_ = self.x_out_resh(x_)
        return x_

    def c_fwd(self, x, c):
        x_ = self.x_in_resh(x)
        c_ = self.c_in_resh(c)
        c_ = self.codeq_linear.c_fwd(x_, c_)
        c_ = self.c_out_resh(c_)
        return c_

    def forward(self, x, c):
        x_ = self.x_fwd(x, c)
        c_ = self.c_fwd(x, c)
        return x_, c_

class CodeqMLP(torch.nn.Module):
    '''
        An MLP layer which updates coloured inputs and is equivariant to permutations of colours.
        It is assumed that colours can have any number of channels associated to them, so that the
        input tensors x, c are in the form:
            x: [num_nodes (or batch_size), in_dim]
            c: [num_nodes (or batch_size), num_colours, colour_in_dim]
    '''
    # TODO: Properly handle bias terms...
    def __init__(self, in_dim, colour_in_dim, emb_dim, colour_emb_dim, multiplier=1, bn=True):
        super(CodeqMLP, self).__init__()
        self.bn = bn
        self.multiplier = multiplier
        self.colour_emb_dim = colour_emb_dim
        self.linear1 = CodeqLinear(in_dim, colour_in_dim, multiplier * emb_dim, multiplier * colour_emb_dim)
        if self.bn:
            self.bn_x = torch.nn.BatchNorm1d(multiplier * emb_dim)
            self.bn_c = torch.nn.BatchNorm1d(multiplier * colour_emb_dim)
        self.linear2 = CodeqLinear(multiplier * emb_dim, multiplier * colour_emb_dim, emb_dim, colour_emb_dim)

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        if self.bn:
            self.bn_x.reset_parameters()
            self.bn_c.reset_parameters()

    def forward(self, x, c):
        x_, c_ = self.linear1(x, c)
        if self.bn:
            x_ = self.bn_x(x_)
            # in order to apply BN to colours we rearrange them all over the first dimension
            # ... and of course we reshape back after the application of BN
            # TODO: discuss this approach
            n, k, d = c_.shape
            assert d == self.multiplier * self.colour_emb_dim
            c_aux = c_.reshape((n*k, d))
            c_aux = self.bn_c(c_aux)
            c_ = c_aux.reshape((n,k,d))
        x_ = F.relu(x_)
        c_ = F.relu(c_)
        x_, c_ = self.linear2(x_, c_)
        return x_, c_

class CodeqSharedMLP(torch.nn.Module):
    '''
        An MLP layer which updates coloured inputs and is equivariant to permutations of colours.
        It is assumed that colours can have any number of channels associated to them.
        Additionally, each input element can come in different colourings, referred to as "samples".
        The layer transform them in parallel, naturally sharing the same parameters, hence the "Shared"
        apposition. Effectively, input tensors x, c are in the form:
            x: [num_nodes (or batch_size), num_samples*in_dim]
            c: [num_nodes (or batch_size), num_samples*num_colours, colour_in_dim]

    '''
    def __init__(self, in_dim, colour_in_dim, emb_dim, colour_emb_dim, num_samples, multiplier=1, bn=True):
        super(CodeqSharedMLP, self).__init__()
        self.bn = bn
        self.multiplier = multiplier
        self.colour_emb_dim = colour_emb_dim
        self.num_samples = num_samples
        self.x_in_resh = Reshaper(in_dim, num_samples, out=False, stacked_samples=True)
        self.x_out_resh = Reshaper(emb_dim, num_samples, out=True, stacked_samples=True)
        self.c_in_resh = ColourReshaper(colour_in_dim, num_samples, out=False)
        self.c_out_resh = ColourReshaper(colour_emb_dim, num_samples, out=True)
        self.codeq_mlp = CodeqMLP(in_dim, colour_in_dim, emb_dim, colour_emb_dim, multiplier=multiplier, bn=bn)

    def reset_parameters(self):
        self.x_in_resh.reset_parameters()
        self.x_out_resh.reset_parameters()
        self.c_in_resh.reset_parameters()
        self.c_out_resh.reset_parameters()
        self.codeq_mlp.reset_parameters()

    def forward(self, x, c):
        x_ = self.x_in_resh(x)
        c_ = self.c_in_resh(c)
        x_, c_ = self.codeq_mlp(x_, c_)
        x_ = self.x_out_resh(x_)
        c_ = self.c_out_resh(c_)
        return x_, c_