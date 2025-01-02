'''
    From the LoG 2022 tutorial, slightly adapted to include other source graphs and 
    path counting as a target
'''

import os.path as osp
import scipy.io as sio
import numpy as np
import torch
import networkx as nx

from collections import Counter
from tqdm import tqdm

from scipy.special import comb
from torch_geometric.data import InMemoryDataset, Data

from constants import PATHS_MAX_LENGTH, DEFAULT_LARGER_RR_PARAMETERS, DATA_PATH

from data_generation_utils import generate_RR_mat_dataset

def count_paths(adjacency_matrix, cutoff):

    # 0. instantiate Counter
    counter = Counter({i: 0 for i in range(1,cutoff+1)})

    # 0. get nx graph
    G = nx.Graph(adjacency_matrix)

    # 1. iterate over each node and count paths towards all other nodes with larger index
    n = G.number_of_nodes()
    for source in range(n):
        targets = list(range(source+1, n))
        paths = nx.all_simple_paths(G, source=source, target=targets, cutoff=cutoff)
        path_lengths = list(map(lambda x: len(x)-1, paths))
        counter.update(path_lengths)

    # 2. arrange into list and return
    return [counter[l] for l in sorted(counter)][1:]  # skip edges

class CountingSubstructures(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, include_paths=False, parameters=None, largest_component=False):
        self.name = name
        self.largest_component = largest_component
        if parameters is None:
            source = "randomgraph"
        else:
            if parameters == 'default_larger':
                parameters = DEFAULT_LARGER_RR_PARAMETERS
            source = "randomgraph"+f"_RR_{parameters}"
            par_list = parameters.strip().split('_')
            assert len(par_list) == 4
            par_list_parsed = list()
            for par_string in par_list:
                vals = par_string.strip().split('-')
                assert len(vals) == 2
                par_list_parsed.append([int(vals[0]), int(vals[1])])
            parameters = par_list_parsed
            self.name += '_'+source  # write the source in the name to set up a different processed dir
        self.parameters = parameters
        self.source = source
        self.include_paths = include_paths
        super(CountingSubstructures, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # Normalize as in GNN-AK
        self.data.y = self.data.y / self.data.y.std(0)

        a = sio.loadmat(osp.join(self.raw_dir, self.raw_file_names[0]))
        self.train_idx = torch.from_numpy(a["train_idx"][0])
        self.val_idx = torch.from_numpy(a["val_idx"][0])
        self.test_idx = torch.from_numpy(a["test_idx"][0])

    @property
    def raw_dir(self):
        return DATA_PATH

    @property
    def raw_file_names(self):
        return [f"{self.source}.mat"]

    @property
    def processed_file_names(self):
        name = "data.pt" if not self.include_paths else f"data_with_paths_up_to_{PATHS_MAX_LENGTH}.pt"
        return name

    @property
    def processed_dir(self):
        name = "processed"
        return osp.join(self.root, self.name, name)

    def download(self):
        # Generate to `self.raw_dir`.
        if self.parameters is not None:
            print(f'Generating dataset with parameters {self.parameters}...')
            generate_RR_mat_dataset(osp.join(self.raw_dir, self.raw_file_names[0]), self.parameters, largest_component=self.largest_component)
        return

    @property
    def num_tasks(self):
        return 1

    def process(self):
        # Read data into huge `Data` list.
        b = self.processed_paths[0]
        a = sio.loadmat(f"./data/{self.source}.mat")
        # list of adjacency matrix
        A = a["A"][0]
        
        data_list = []
        for i in tqdm(range(len(A))):
            a = A[i]
            A2 = a.dot(a)
            A3 = A2.dot(a)
            A4 = A3.dot(a)
            tri = np.trace(A3) / 6
            tailed = ((np.diag(A3) / 2) * (a.sum(0) - 2)).sum()
            cyc4 = 1 / 8 * (np.trace(A3.dot(a)) + np.trace(A2) - 2 * A2.sum())  # 1/8[∑i=1nλ4i−∑i∈V(G)di(2di−1)],           
            cyc5 = 1 / 10 * (np.trace(A4.dot(a)) - 5 * np.trace(A3) - 5 * (np.sum(np.diag(A3) * (np.sum(a, axis=1) - 2))))
            cus = a.dot(np.diag(np.exp(-a.dot(a).sum(1)))).dot(a).sum()  

            deg = a.sum(0)
            star = 0
            for j in range(a.shape[0]):
                star += comb(int(deg[j]), 3)

            if self.include_paths:
                num_paths = count_paths(a, PATHS_MAX_LENGTH)
                assert len(num_paths) == PATHS_MAX_LENGTH - 1, num_paths
                expy = torch.tensor([[tri, tailed, star, cyc4, cyc5]+num_paths])
            else:
                expy = torch.tensor([[tri, tailed, star, cyc4, cyc5]])

            E = np.where(A[i] > 0)
            edge_index = torch.Tensor(np.vstack((E[0], E[1]))).type(torch.int64)
            x = torch.ones(A[i].shape[0], 1)
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def separate_data(self, seed):
        return {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}
