'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Credit: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
VAE Credit: https://github.com/AntixK/PyTorch-VAE/tree/a6896b944c918dd7030e7d795a8c13e5c6345ec7
Contrastive Loss: https://lilianweng.github.io/posts/2021-05-31-contrastive/
CLIP train: https://github.com/openai/CLIP/issues/83

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
	Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import clip
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"


class CLIP_AE_Encode(nn.Module):
	def __init__(self, hidden_dim, latent_dim, isAE=False):
		super(CLIP_AE_Encode, self).__init__()
		# Build Encoder
		self.fc1 = nn.Linear(512, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, latent_dim)
		self.relu = nn.ReLU(inplace=True)

		if isAE:
			self.filter = nn.Parameter(torch.ones((512)))
		else:
			self.filter = nn.Parameter(torch.rand((512)))

	def forward(self, clip_model, images):
		with torch.no_grad():
			emb = clip_model.encode_image(images).float()
		out = emb * self.filter
		out = self.relu(self.fc1(out))
		z = self.fc2(out)

		return z