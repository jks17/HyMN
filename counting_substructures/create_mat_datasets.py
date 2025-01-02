import os.path as osp
import scipy.io as sio
import numpy as np
import torch

from scipy.special import comb
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx
import random

# imports for dgl dataset
import dgl
from dgl.data.utils import load_graphs

from constants import DATA_PATH

# seeds
seed = 0
np.random.seed(seed)
random.seed(seed)
random_state = np.random.RandomState(seed)

# create random regular dataset and ER dataset .mat files from .bin files from https://github.com/leichen2018/GNN-Substructure-Counting/tree/main/synthetic/data

dataset_path = f"{DATA_PATH}"
dataset_source_path = f"{dataset_path}/data"
dataset_names = ['dataset1', 'dataset2'] # names of the .bin files
save_file_names = ['randomgraphsER.mat', 'randomgraphsRR.mat'] # what to name the .mat files for the dataset names

for dataset_idx, name in enumerate(dataset_names):
    glist, all_labels = load_graphs(dataset_source_path + '/' + name + '.bin')
    train_idx = []
    with open(dataset_source_path + '/' + name + '_train.txt', "r") as f:
        for line in f:
            train_idx.append(int(line.strip()))
    val_idx = []
    with open(dataset_source_path + '/' + name + '_val.txt', "r") as f:
        for line in f:
            val_idx.append(int(line.strip()))
    test_idx = []
    with open(dataset_source_path + '/' + name + '_test.txt', "r") as f:
        for line in f:
            test_idx.append(int(line.strip()))
            
    # generate A, F keys (adjacency matrices and labels) for .mat file
    adjacency_matrices = []
    F = []

    for idx, graph in enumerate(glist):
        labels = [float(all_labels['tailed_triangle'][idx]), float(all_labels['star'][idx]), float(all_labels['triangle'][idx]), float(all_labels['chordal_cycle'][idx])]
        adj = np.array(graph.adj(scipy_fmt='coo').todense())
        adjacency_matrices.append(adj)
        F.append(labels)
        
    mat_dict = b = {'A': [adjacency_matrices], 'F': [F], 'train_idx': np.array([train_idx]), 'val_idx': np.array([val_idx]), 'test_idx': np.array([test_idx])}
    
    sio.savemat(dataset_path + '/' + save_file_names[dataset_idx], mat_dict)
    
# Test 1: 'randomgraphsRR.mat' == original 'randomgraphs.mat' 
# Test 2: edges in 'randomgraphsER.mat' are added with = 0.3 (approximately) and all have 10 nodes 

   
# create a larger dataset of ER graphs and RR graphs

# RR graphs
save_file_name = 'randomgraphsRR_larger.mat'
parameters = [[20, 5], [30, 5], [40, 5], [60, 5]]
number_of_graphs = 5000

random_regular_graphs = []
for i in range(number_of_graphs):
    m, d = random.choice(parameters) # choose parameters uniformly to create regular graph
    G = nx.random_regular_graph(d, m, seed=random_state)
    
    edges_to_remove = random.sample(list(G.edges()), m) # remove m edges
    for edge in edges_to_remove:
        G.remove_edge(*edge)
        
    random_regular_graphs.append(G)
    
adjacency_matrices = [np.asarray(nx.adjacency_matrix(G).todense()) for G in random_regular_graphs]
l = [[0, 0, 0, 0] for G in random_regular_graphs] # make the labels all zeros as we calculate in the loading function

# use the same training, val and test as original smaller graphs
mat_dict = {'A': [adjacency_matrices], 'F': [l], 'train_idx': np.array([train_idx]), 'val_idx': np.array([val_idx]), 'test_idx': np.array([test_idx])}

sio.savemat(dataset_path + '/' + save_file_name, mat_dict)
    
    
    
# Test 3: Loading into pytorch and counting substructures works with original code
# Test 4: Graphs are uniformly split roughly across number of nodes
    
    

# ER graphs
save_file_name = 'randomgraphsER_larger.mat'
n = 50 # number nodes
p = 0.3
num_graphs = 5000


er_graphs = [nx.erdos_renyi_graph(n, p, seed=random_state) for i in range(num_graphs)]
adjacency_matrices = [np.asarray(nx.adjacency_matrix(G).todense()) for G in er_graphs]
l = [[0, 0, 0, 0] for G in er_graphs] # again we don't know label information


# use the same training, val and test as original smaller graphs
mat_dict = {'A': [adjacency_matrices], 'F': [l], 'train_idx': np.array([train_idx]), 'val_idx': np.array([val_idx]), 'test_idx': np.array([test_idx])}

sio.savemat(dataset_path + '/' + save_file_name, mat_dict)


# Test 5: (Repeated behaviour - are the same datasets generated on every run)
