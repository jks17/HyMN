import networkx as nx
import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils.undirected import is_undirected
from torch_geometric.utils import to_networkx, to_dense_adj, get_laplacian
from numpy.linalg import eigvals
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj, scatter)
from torch_geometric.utils.num_nodes import maybe_num_nodes

from sklearn.manifold import MDS
import math

import numpy as np
from sklearn.metrics import pairwise_distances


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


def centrality_posenc(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    P = to_dense_adj(edge_index, max_num_nodes=num_nodes)
    rws = []
    # Efficient way if ksteps are a consecutive sequence (true in this case)
    Pk = P.clone().detach().matrix_power(min(ksteps))
    for k in range(min(ksteps), max(ksteps) + 1):
        if k == 1:
            rws.append((Pk.sum(dim=1)/math.factorial(k)))
        else:
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1)/math.factorial(k))
        Pk = Pk @ P
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


class RandomSampling(BaseTransform):
        def __init__(self, seed=0):
                self.seed = seed
        def __call__(self, data):
            assert data.num_nodes > 0
            assert data.edge_index.shape[1] > 0
            assert is_undirected(data.edge_index)
            data.community = torch.randperm(data.num_nodes)
            return data
            
            
class SubgraphTransform(BaseTransform):

        def __init__(self, seed=0):
                self.seed = seed

        def __call__(self, data):
                assert data.num_nodes > 0
                assert data.edge_index.shape[1] > 0
                assert is_undirected(data.edge_index)
                G = to_networkx(data, to_undirected=True)
                centralities = centrality_posenc(ksteps=range(1,21),
                                          edge_index=data.edge_index,
                                          num_nodes=data.num_nodes)
                order = torch.argsort(centralities.sum(dim=1))
                #order = torch.argsort(-centralities.sum(dim=1))
                data.community = order

                return data 
        
        