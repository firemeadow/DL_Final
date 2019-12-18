import torch
import torch.nn as nn


class Generator(nn.Module):
	
	def __init__(self, num_attr, hidden_layer_size=500):
		super(Generator, self).__init__()
		self.num_attr = num_attr
		self.hidden_layer_size = hidden_layer_size
		self.L1 = nn.LSTM(num_attr, hidden_layer_size, 1)
		self.L2 = nn.Linear(hidden_layer_size, 1)
		
		self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).float(), torch.zeros(1,1,self.hidden_layer_size).float())
		
	def forward(self, x):
		x = x.view(1, 1, len(x))
		lstm_out, self.hidden_cell = self.L1(x.float(), self.hidden_cell)
		out = self.L2(lstm_out.view(len(x), -1))
		return out[-1]