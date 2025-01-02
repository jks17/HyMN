import torch

from constants import COLOURED_MODELS, SHARED_COLOURED_MODELS, VERBOSE_COLOURED_MODELS, CODEQ_MODELS

from nn import SharedLinear, SharedMLP
from readout import (SumReadout, MeanReadout, MaxReadout, SumJKReadout, MeanJKReadout, MaxJKReadout,
    MeanAveraging, SumAveraging, MaxAveraging, SharedJKReadout, ColourReadout, ColourJKReadout,
    AdaptiveMeanAveraging, AdaptiveSumAveraging, AdaptiveMaxAveraging)
from model import GNN, ColourCatSharedGNN, CodeqSharedGNN
from verbose_model import VerboseColourCatGNN

from colour import Painter
from verbose_painter import VerbosePainter
from adaptive_colour import build_adaptive_painter

def get_readout_module(args):
    if not args.jk:
        if args.readout == 'sum':
            readout_module = SumReadout()
        elif args.readout == 'mean':
            readout_module = MeanReadout()
        elif args.readout == 'max':
            readout_module = MaxReadout()
        else:
            raise ValueError(f"Unsupported readout '{args.readout}'")
    else:
        if args.model in SHARED_COLOURED_MODELS:
            if args.readout == 'sum':
                readout = SumReadout()
            elif args.readout == 'mean':
                readout = MeanReadout()
            elif args.readout == 'max':
                readout = MaxReadout()
            else:
                raise ValueError(f"Unsupported readout '{args.readout}'")
            readout_module = SharedJKReadout(readout, args.out_dim, args.emb_dim, args.num_layers, args.num_samples)
        else:
            if args.readout == 'sum':
                readout_module = SumJKReadout()
            elif args.readout == 'mean':
                readout_module = MeanJKReadout()
            elif args.readout == 'max':
                readout_module = MaxJKReadout()
            else:
                raise ValueError(f"Unsupported readout '{args.readout}'")
    # verbose
    msg = f"Readout {readout_module} instantiated.\n"
    print(msg)
    return readout_module

def get_colour_readout_module(args):
    if args.readout == 'sum':
        readout = SumReadout()
    elif args.readout == 'mean':
        readout = MeanReadout()
    elif args.readout == 'max':
        readout = MaxReadout()
    else:
        raise ValueError(f"Unsupported readout '{args.readout}'")
    if not args.colour_jk:
        colour_readout_module = ColourReadout(readout)
    else:
        colour_readout_module = ColourJKReadout(readout)
    # verbose
    msg = f"Colour readout {colour_readout_module} instantiated.\n"
    print(msg)
    return colour_readout_module

def get_prediction_module(args):
    predictor_in_dim = args.emb_dim if not args.jk else (args.emb_dim * args.num_layers + 1)
    if args.model in CODEQ_MODELS:
        predictor_colour_in_dim = args.colour_emb_dim if not args.colour_jk else (args.colour_emb_dim * args.num_layers + 1)
        predictor_in_dim += predictor_colour_in_dim
    if args.predictor == 'id':
        prediction_module = torch.nn.Identity()
    elif args.predictor == 'linear':
        if args.model in SHARED_COLOURED_MODELS:
            prediction_module = SharedLinear(predictor_in_dim, args.out_dim, args.num_samples)
        else:
            prediction_module = torch.nn.Linear(predictor_in_dim, args.out_dim)
    elif args.predictor == 'mlp':
        if args.model in SHARED_COLOURED_MODELS:
            prediction_module = SharedMLP(predictor_in_dim, args.out_dim, args.num_samples, multiplier=(args.multiplier*predictor_in_dim), bn=args.bn)
        else:
            prediction_module = torch.nn.Sequential(
                torch.nn.Linear(predictor_in_dim, args.multiplier * predictor_in_dim),
                torch.nn.BatchNorm1d(args.multiplier * predictor_in_dim) if args.bn else torch.nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Linear(args.multiplier * predictor_in_dim, args.out_dim))
    else:
        raise ValueError(f"Unsupported predictor '{args.predictor}'")
    # verbose
    msg = f"Predictor {prediction_module} instantiated.\n"
    print(msg)
    return prediction_module

def get_averaging_module(args):
    if args.model in VERBOSE_COLOURED_MODELS:
        if args.averaging == 'sum':
            averaging_module = SumReadout()
        elif args.averaging == 'mean':
            averaging_module = MeanReadout()
        elif args.averaging == 'max':
            averaging_module = MaxReadout()
        else:
            raise ValueError(f"Unsupported averaging '{args.averaging}'")
    elif args.model in SHARED_COLOURED_MODELS:
        if args.subsampling_ratio == 0.0:
            if args.averaging == 'sum':
                averaging_module = SumAveraging(args.out_dim, args.num_samples)
            elif args.averaging == 'mean':
                averaging_module = MeanAveraging(args.out_dim, args.num_samples)
            elif args.averaging == 'max':
                averaging_module = MaxAveraging(args.out_dim, args.num_samples)
            else:
                raise ValueError(f"Unsupported averaging '{args.averaging}'")
        else:
            if args.averaging == 'sum':
                averaging_module = AdaptiveSumAveraging(args.out_dim)
            elif args.averaging == 'mean':
                averaging_module = AdaptiveMeanAveraging(args.out_dim)
            elif args.averaging == 'max':
                averaging_module = AdaptiveMaxAveraging(args.out_dim)
            else:
                raise ValueError(f"Unsupported averaging '{args.averaging}'")
    else:
        raise AssertionError
    # verbose
    msg = f"Averaging {averaging_module} instantiated.\n"
    print(msg)
    return averaging_module


def get_model(args):

    # instantiate readout module
    readout_module = get_readout_module(args)

    # instantiate prediction module
    prediction_module = get_prediction_module(args)

    # instantiate averaging module for coloured models
    if args.model in COLOURED_MODELS:
        averaging_module = get_averaging_module(args)

    # instantiate colour readout module needed for codeq models
    if args.model in CODEQ_MODELS:
        colour_readout_module = get_colour_readout_module(args)

    # instantiate model
    if args.model == 'ColourCatSharedGNN':
        model = ColourCatSharedGNN(args.num_layers, args.in_dim, args.emb_dim, args.num_colours, args.num_samples, readout_module, prediction_module, averaging_module, 
            layer=args.layer, residual=args.residual, multiplier=args.multiplier, inject_colours=args.inject_colours, bn=args.bn, sample_aggregation=args.sample_aggregation, bn_between_convs=args.bn_between_convs)
    elif args.model == 'CodeqSharedGNN':
        model = CodeqSharedGNN(args.num_layers, args.in_dim, args.emb_dim, args.num_colours if args.encoding_scheme == 'onehot_2d' else 1, args.colour_emb_dim, args.num_colours, 
            args.num_samples, readout_module, colour_readout_module, prediction_module, averaging_module, layer=args.layer, residual=args.residual,
            colour_picker_agg=args.colour_picker_aggregator, multiplier=args.multiplier, bn=args.bn, picker_b4_readout=args.picker_b4_readout)
    elif args.model == 'VerboseColourCatGNN':
        model = VerboseColourCatGNN(args.num_layers, args.in_dim, args.emb_dim, args.num_colours, readout_module, prediction_module, averaging_module, layer=args.layer,
            residual=args.residual, multiplier=args.multiplier, inject_colours=args.inject_colours, bn=args.bn)
    else:
        model = GNN(args.num_layers, args.in_dim, args.emb_dim, readout_module, prediction_module, layer=args.layer, residual=args.residual, multiplier=args.multiplier, bn=args.bn, bn_between_convs=args.bn_between_convs)

    return model

def get_painters(args, device):
    # set up painters
    train_painter = None
    train_eval_painter = None
    val_painter = None
    test_painter = None
    if args.subsampling_ratio == 0.0:
        painter_class = Painter if not args.verbose_data_pipe else VerbosePainter
        if args.num_colours > 0:
            train_painter = painter_class(args.seed, args.num_colours, args.nodes_to_colour, device=device, colouring_scheme=args.colouring_scheme, encoding=args.encoding_scheme, 
                num_samples=args.num_samples)
            train_eval_painter = painter_class(args.seed, args.num_colours, args.nodes_to_colour, device=device, colouring_scheme=args.colouring_scheme, encoding=args.encoding_scheme, 
                num_samples=args.num_samples)
            val_painter = painter_class(args.seed, args.num_colours, args.nodes_to_colour, device=device, colouring_scheme=args.colouring_scheme, encoding=args.encoding_scheme, 
                num_samples=args.num_samples)
            test_painter = painter_class(args.seed, args.num_colours, args.nodes_to_colour, device=device, colouring_scheme=args.colouring_scheme, encoding=args.encoding_scheme, 
                num_samples=args.num_samples)
    else:
        train_painter = build_adaptive_painter(args.colouring_scheme, args.seed, device, args.num_samples, args.subsampling_ratio, num_colours=args.num_colours,
            encoding_scheme=args.encoding_scheme)
        train_eval_painter = build_adaptive_painter(args.colouring_scheme, args.seed, device, args.num_samples, args.subsampling_ratio, num_colours=args.num_colours,
            encoding_scheme=args.encoding_scheme)
        val_painter = build_adaptive_painter(args.colouring_scheme, args.seed, device, args.num_samples, args.subsampling_ratio, num_colours=args.num_colours,
            encoding_scheme=args.encoding_scheme)
        test_painter = build_adaptive_painter(args.colouring_scheme, args.seed, device, args.num_samples, args.subsampling_ratio, num_colours=args.num_colours,
            encoding_scheme=args.encoding_scheme)

    # verbose
    msg = f"Painters instantiated with {args.num_colours} colours, {args.num_samples} samples per input graph, {args.colouring_scheme}"
    msg += f" scheme and {args.encoding_scheme} encoding.\n"
    print(msg)
    return train_painter, train_eval_painter, val_painter, test_painter
