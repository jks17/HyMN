import torch

from graphgps.layer.nn import Reshaper
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class SumReadout(torch.nn.Module):

	def __init__(self):
		super(SumReadout, self).__init__()

	def reset_parameters(self):
		return

	def forward(self, h, index):
		if isinstance(h, list):
			h = h[-1]
		h = global_add_pool(h, index)
		return h


class MeanReadout(torch.nn.Module):

	def __init__(self):
		super(MeanReadout, self).__init__()

	def reset_parameters(self):
		return

	def forward(self, h, index):
		if isinstance(h, list):
			h = h[-1]
		h = global_mean_pool(h, index)
		return h


class MaxReadout(torch.nn.Module):

	def __init__(self):
		super(MaxReadout, self).__init__()

	def reset_parameters(self):
		return

	def forward(self, h, index):
		if isinstance(h, list):
			h = h[-1]
		h = global_max_pool(h, index)
		return h


class SumJKReadout(torch.nn.Module):

	def __init__(self):
		super(SumJKReadout, self).__init__()

	def reset_parameters(self):
		return

	def forward(self, h_list, index):
		assert isinstance(h_list, list)
		h = torch.cat(h_list, dim=1)
		h = global_add_pool(h, index)
		return h


class MeanJKReadout(torch.nn.Module):

	def __init__(self):
		super(MeanJKReadout, self).__init__()

	def reset_parameters(self):
		return

	def forward(self, h_list, index):
		assert isinstance(h_list, list)
		h = torch.cat(h_list, dim=1)
		h = global_mean_pool(h, index)
		return h


class MaxJKReadout(torch.nn.Module):

	def __init__(self):
		super(MaxJKReadout, self).__init__()

	def reset_parameters(self):
		return

	def forward(self, h_list, index):
		assert isinstance(h_list, list)
		h = torch.cat(h_list, dim=1)
		h = global_max_pool(h, index)
		return h


class MeanAveraging(torch.nn.Module):

	def __init__(self, emb_dim, num_samples):
		super(MeanAveraging, self).__init__()
		self.emb_dim = emb_dim
		self.num_samples = num_samples

	def reset_parameters(self):
		return

	def forward(self, h):
		assert h.ndim == 2
		h = h.reshape(-1, self.num_samples, self.emb_dim)
		return h.mean(1)


class SumAveraging(torch.nn.Module):

	def __init__(self, emb_dim, num_samples):
		super(SumAveraging, self).__init__()
		self.emb_dim = emb_dim
		self.num_samples = num_samples

	def reset_parameters(self):
		return

	def forward(self, h):
		assert h.ndim == 2
		h = h.reshape(-1, self.num_samples, self.emb_dim)
		return h.sum(1)


class MaxAveraging(torch.nn.Module):

	def __init__(self, emb_dim, num_samples):
		super(MaxAveraging, self).__init__()
		self.emb_dim = emb_dim
		self.num_samples = num_samples

	def reset_parameters(self):
		return

	def forward(self, h):
		assert h.ndim == 2
		h = h.reshape(-1, self.num_samples, self.emb_dim)
		return h.max(1).values


class SharedJKReadout(torch.nn.Module):

	def __init__(self, readout, in_dim, emb_dim, num_layers, num_samples):
		super(SharedJKReadout, self).__init__()
		self.readout = readout
		self.out_dim = in_dim+num_layers*emb_dim
		self.first_in_resh = Reshaper(in_dim, num_samples, out=False, stacked_samples=True)
		self.in_resh = Reshaper(emb_dim, num_samples, out=False, stacked_samples=True)
		self.out_resh = Reshaper(self.out_dim, num_samples, out=True, stacked_samples=True)

	def reset_parameters(self):
		self.readout.reset_parameters()
		self.in_resh.reset_parameters()
		self.out_resh.reset_parameters()

	def forward(self, x_list, index):
		assert isinstance(x_list, list)
		x_list_ = list()  # [[num_samples*num_nodes, num_feats], ...]
		for i, x in enumerate(x_list):
			if i == 0:
				x_list_.append(self.first_in_resh(x))
			else:
				x_list_.append(self.in_resh(x))
		x_ = torch.cat(x_list_, dim=1)  # [num_samples*num_nodes, in_dim+num_layers*emb_dim]
		assert self.out_dim == x_.shape[1]
		x = self.out_resh(x_)  # [num_nodes, num_samples*(in_dim+num_layers*emb_dim)]
		return self.readout(x, index)


class ColourReadout(torch.nn.Module):

	def __init__(self, readout):
		super(ColourReadout, self).__init__()
		self.readout = readout

	def reset_parameters(self):
		self.readout.reset_parameters()

	def forward(self, c, index):
		if isinstance(c, list):
			c = c[-1]
		emb_dim = c.shape[2]
		c_flat = c.flatten(start_dim=-2, end_dim=-1)
		y_flat = self.readout(c_flat, index)
		return y_flat.unflatten(-1, (-1, emb_dim))


class ColourJKReadout(torch.nn.Module):

	def __init__(self, readout):
		super(ColourJKReadout, self).__init__()
		self.readout = readout

	def reset_parameters(self):
		self.readout.reset_parameters()

	def forward(self, c_list, index):
		assert isinstance(c_list, list)
		c = torch.cat(c_list, dim=2)
		emb_dim = c.shape[2]
		assert c.ndim == 3
		c_flat = c.flatten(start_dim=-2, end_dim=-1)
		y_flat = self.readout(c_flat, index)
		return y_flat.unflatten(-1, (-1, emb_dim))


class AdaptiveMeanAveraging(torch.nn.Module):

	def __init__(self, emb_dim):
		super(AdaptiveMeanAveraging, self).__init__()
		self.emb_dim = emb_dim

	def reset_parameters(self):
		return

	def forward(self, h, indices, batch):
		assert h.ndim == 2
		h = h.reshape(-1, self.emb_dim)
		valid_h = torch.index_select(h,0,indices.to(h.device))
		return global_mean_pool(valid_h.to(h.device),batch.to(h.device))


class AdaptiveSumAveraging(torch.nn.Module):

	def __init__(self, emb_dim):
		super(AdaptiveSumAveraging, self).__init__()
		self.emb_dim = emb_dim

	def reset_parameters(self):
		return

	def forward(self, h, indices, batch):
		assert h.ndim == 2
		h = h.reshape(-1, self.emb_dim)
		valid_h = torch.index_select(h,0,indices.to(h.device))
		return global_add_pool(valid_h.to(h.device),batch.to(h.device))


class AdaptiveMaxAveraging(torch.nn.Module):

	def __init__(self, emb_dim):
		super(AdaptiveMaxAveraging, self).__init__()
		self.emb_dim = emb_dim

	def reset_parameters(self):
		return

	def forward(self, h, indices, batch):
		assert h.ndim == 2
		h = h.reshape(-1, self.emb_dim)
		valid_h = torch.index_select(h,0,indices.to(h.device))
		return global_max_pool(valid_h.to(h.device),batch.to(h.device))