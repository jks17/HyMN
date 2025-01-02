import subprocess
import argparse

from constants import COLOURED_MODELS, VERBOSE_COLOURED_MODELS, CODEQ_MODELS, DSS_LAYERS

#def get_git_revision_hash() -> str:
#    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

#def get_git_revision_short_hash() -> str:
#    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_colour_args(args):
    if args.model in COLOURED_MODELS:
        assert args.averaging != 'none'
        assert args.num_samples > 0
        assert args.colouring_scheme != 'none'
        assert args.subsampling_ratio >= 0.0 and args.subsampling_ratio <= 1.0
        if args.subsampling_ratio == 0.0:
            assert args.num_colours > 0
            assert args.encoding_scheme != 'none'
        else:
            assert args.model not in VERBOSE_COLOURED_MODELS
            assert args.model not in CODEQ_MODELS
        if args.model in CODEQ_MODELS:
            assert args.colour_readout != 'none'
            assert args.colour_picker_aggregator != 'none'
            assert args.colour_emb_dim > 0
        if args.model in VERBOSE_COLOURED_MODELS:
            assert args.verbose_data_pipe
        else:
            assert not args.verbose_data_pipe
        if args.layer in DSS_LAYERS:
            assert args.sample_aggregation != 'none'
        if args.colouring_scheme in ['uniform', 'spread', 'shifted_spread_fiedler', 'indexcentralitymulti', 'indexcentralityelbow']:
            assert args.encoding_scheme in ['onehot', 'onehot_2d']
        elif args.colouring_scheme == 'spread_fiedler':
            assert args.encoding_scheme == 'onehot'
            assert args.num_samples == 1
        elif args.colouring_scheme == 'fiedler_vector':
            assert args.num_samples == 1
            assert args.num_colours == 1
            assert args.encoding_scheme == 'identity'
        elif args.colouring_scheme == 'index':
            assert args.encoding_scheme == 'identity'
            assert args.num_colours == 1
        elif args.colouring_scheme == 'higherorder':
            assert args.encoding_scheme in ['onehot', 'onehot_2d']
        elif args.colouring_scheme == 'pairdistance':
            assert args.encoding_scheme in ['onehot', 'onehot_2d']
            assert args.num_colours == 3
        else:
            assert args.encoding_scheme == 'identity'
    return
