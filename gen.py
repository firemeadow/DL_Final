import torch
import torch.nn as nn


class Generator(nn.Module):
	
	def __init__(self, num_attr, minibatch_size):
		super(Generator, self).__init__()
		self.num_attr = num_attr
		self.minibatch_size = minibatch_size
		
		self.L1 = nn.Linear(num_attr, 500)
		self.L2 = nn.Linear(500, 1)
		
	def forward(self, x):
		num_days = len(x)
		num_batches = num_days - self.minibatch_size
		out = torch.tensor([1,1])
		compare = out
		for i in range(num_batches):
			batch_min = i
			batch_max = i + self.minibatch_size
			temp_x = x[batch_min:batch_max]
			temp_x = self.L1(temp_x)
			temp_x = self.L2(temp_x)
			if not torch.equal(out, compare):
				out = torch.cat([out, temp_x], 0)
			else:
				out = temp_x
		return out