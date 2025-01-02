import numpy as np

import torch
import torch.nn.functional as F

from colour import Painter
from torch_geometric.data import Data, Batch

# TODO add handling of edge attributes
class VerbosePainter(Painter):
	'''
		A painter which replicates the input graph many times, once for each sampled colouring.
		Additional auxiliary variables are inject to enable proper averaging over samples.
	'''

	def colour_data(self, data, colours):
		'''
			Auxiliary routine which augments data objects with tensorial representations of colours.
			Arguments
				`data`: the PyG data object to augment with colour representations
				`colours`: sampled colours
		'''
		assert colours.shape[0] == data.num_nodes
		if self.encoding == 'onehot':
			assert colours.ndim == 1
			data.c = F.one_hot(colours, self.k).to(torch.get_default_dtype())
		elif self.encoding == 'identity':
			data.c = colours.to(torch.get_default_dtype())
		else:
			raise NotImplementedError
		return data

	def paint(self, batch):
		'''
			Auxiliary routine which assigns colours to batch objects in an efficient way.
			Effectively, each batch is replicated `num_samples` times and each graph in
			each batch replica is assigned a different colouring.
			The correspondence between colourings and input graphs is maintained via the
			attributes:
				* `node2graph` – length num_nodes*num_samples; maps each coloured node to the original graph
				* `colouring2graph` – length num_graphs*num_samples; maps each coloured graph to the original graph
			Arguments
				`batch`: the PyG batch object to paint
				`colouring`: the colouring object that can generate colours
		'''
		n = batch.num_nodes * self.num_samples
		replicates = list()
		for _ in range(self.num_samples):
			replicates.extend(batch.to_data_list())
		new_batch = Batch.from_data_list(replicates)
		new_batch.y = batch.y
		new_batch.num_graphs = batch.num_graphs * self.num_samples
		colours = self.colouring.colours(n)
		coloured_batch = self.colour_data(new_batch, colours)
		coloured_batch.colouring2graph = torch.arange(batch.num_graphs).repeat(self.num_samples)
		coloured_batch.node2graph = batch.batch.repeat(self.num_samples)
		return coloured_batch