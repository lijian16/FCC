import torch
import torch.nn.functional as F
import torch.nn as nn 
import sys
import time
import os 
import shutil
import math
from modules import GAP


def fcc(feature, label, num_classes, cfg, **kwargs):
	"""
	Feature clusters compression (FCC)
	gamma: the hyper-parameter for setting scaling factor tau.
	c_type: compression type:
				'edc' is equal difference compression.
	"""

	c_type = cfg.FCC.C_TYPE
	batch_size = feature.shape[0]

	# compressing feature
	if c_type == 'edc':
		new_features = equal_diff_compress(batch_size, feature, label, num_classes, cfg.FCC.GAMMA)

	else:
		raise Exception('Error compression type.')

	return new_features

def equal_diff_compress(n, feature, label, num_classes, gamma):
	# setting scaling factor tau
	tau = []
	for k in range(num_classes):
		tau.append(round((1 + gamma - k*(gamma/num_classes)),2))

	raw_shape = feature.shape

	tau_batch = []
	for j in range(n):
		tau_batch.append(tau[label[j]])
	tau_batch = torch.tensor(tau_batch).cuda()

	tau_batch = tau_batch.view(n, 1)
	feature = feature.view(n, -1)
	
	new_features = torch.mul(feature, tau_batch)
	new_features = new_features.view(raw_shape)

	return new_features






	
