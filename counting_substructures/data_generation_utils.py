import networkx as nx
import numpy as np

from scipy import io as sio
from constants import SUBGRAPH_NUM_GRAPHS, SUBGRAPH_SPLIT_PATH

def generate_RR_mat_dataset(save_path, parameters, largest_component=False):

    # default: [[20, 5], [30, 5], [40, 5], [60, 4]]; see `constants.py`
    random_state = np.random.RandomState(0)
    random_regular_graphs = []
    for i in range(SUBGRAPH_NUM_GRAPHS):
        m, d = parameters[random_state.choice(len(parameters))]  # choose parameters uniformly to create regular graph
        G = nx.random_regular_graph(d, m, seed=random_state)
        
        edges = list(G.edges())
        edges_to_remove = [edges[i] for i in random_state.choice(len(edges),m,replace=False)]  # remove m edges
        for edge in edges_to_remove:
            G.remove_edge(*edge)

        if largest_component:
            largest_cc = max(nx.connected_components(G), key=len)
            G_lcc = nx.convert_node_labels_to_integers(G.subgraph(largest_cc).copy())
            random_regular_graphs.append(G_lcc)
        else:
            random_regular_graphs.append(G)
        
    adjacency_matrices = [np.asarray(nx.adjacency_matrix(G).todense()) for G in random_regular_graphs]
    l = [[0, 0, 0, 0] for G in random_regular_graphs] # make the labels all zeros as we calculate in the loading function

    # use the same training, val and test as original smaller graphs
    train_idx = []
    with open(SUBGRAPH_SPLIT_PATH+'_train.txt', "r") as f:
        for line in f:
            train_idx.append(int(line.strip()))
    val_idx = []
    with open(SUBGRAPH_SPLIT_PATH+'_val.txt', "r") as f:
        for line in f:
            val_idx.append(int(line.strip()))
    test_idx = []
    with open(SUBGRAPH_SPLIT_PATH+'_test.txt', "r") as f:
        for line in f:
            test_idx.append(int(line.strip()))
    mat_dict = {'A': [adjacency_matrices], 'F': [l], 'train_idx': np.array([train_idx]), 'val_idx': np.array([val_idx]), 'test_idx': np.array([test_idx])}
    # NB: the following line throws a warning, due to the fact that, internally, the routine packs `[adjacency_matrices]` into a ragged array;
    # this is due to the fact that matrices may have different sizes. We add this extra dimension just for compatibility with the code in other
    # repositories (see e.g. `https://github.com/LingxiaoShawn/GNNAsKernel/blob/479dee1ab4172a02a4d236c30bdf54f3691f308f/core/data.py#L73`).
    # In any case, with this code, we manage to exactly replicate the mat file provided in the same repository at:
    # `https://github.com/LingxiaoShawn/GNNAsKernel/blob/479dee1ab4172a02a4d236c30bdf54f3691f308f/data/subgraphcount/raw/randomgraph.mat`
    # starting from the loading code and original binary in the seminal repository:
    # `https://github.com/leichen2018/GNN-Substructure-Counting/blob/83efe64aa59e9132cfc34085d680f39cd4f18324/synthetic/data/dataset2.bin`.
    # Let us finally note that a possible fix would be to replace `[adjacency_matrices]` with `np.array([adjacency_matrices], dtype=object)`.
    sio.savemat(save_path, mat_dict)

    return
