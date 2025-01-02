DATA_PATH = './data/'

ZINC_NUM_ATOMS = 21
ZINC_NUM_BONDS = 4

PATHS_MAX_LENGTH = 5
SUBGRAPH_NUM_GRAPHS = 5000
DEFAULT_LARGER_RR_PARAMETERS = '20-5_30-5_40-5_60-4'
SUBGRAPH_COUNT_TASKS = {
        'tri': 0,
        'tailed': 1,
        'star': 2,
        'cyc4': 3,
        'cyc5': 4} #'cus': 4}
SUBGRAPH_COUNT_TASKS.update({f'path{l}': 5+l-2 for l in range(2,PATHS_MAX_LENGTH+1)})
SUBGRAPH_SPLIT_PATH = f'{DATA_PATH}/dataset2'

DSS_LAYERS = [
	# layers which adopt the DSS approach
	'dss_gin',
	'colourcat_dss_gin']

COLOURED_MODELS = [
	# models which make use of colourings
	'ColourCatGNN',
	'ColourCatSharedGNN',
	'CodeqSharedGNN',
	'VerboseColourCatGNN']

VERBOSE_COLOURED_MODELS = [
	# coloured models where different samples are simply considered
	# as further additional graphs in the batch
	'VerboseColourCatGNN']

SHARED_COLOURED_MODELS = [
	# coloured models where different samples are handled together
	# in parallel by layers which act in a shared fashion
	'ColourCatGNN',
	'ColourCatSharedGNN',
	'CodeqSharedGNN']

CODEQ_MODELS = [
	# coloured models which process colours in an equivariant manner
	'CodeqSharedGNN']

assert (set(VERBOSE_COLOURED_MODELS) | set(SHARED_COLOURED_MODELS)) == set(COLOURED_MODELS)

OUT_DIMS = {
	'disambiguation': 2,
	'subgraph_count': 1,
	'zinc': 1,
        'ppa': 37}

IN_DIMS = {
	'disambiguation': 1,
	'subgraph_count': 1,
	'zinc': 1,
        'ppa': 1}

MAX_SIZES = {
	'subgraph_count_None': 40,
	'subgraph_count_60-5_60-5_60-5_60-5': 60,
	}
