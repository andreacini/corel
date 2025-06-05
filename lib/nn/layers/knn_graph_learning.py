import torch
from einops import repeat
from torch import nn, inference_mode

from torch import Tensor

from lib.nn.layers.samplers.subset import StraightThroughSubsetSampler
from lib.nn.utils import adj_to_fc_edge_index
from tsl.nn.layers import NodeEmbedding
import tsl
from tsl.ops.connectivity import adj_to_edge_index


class AdjEmb(nn.Module):
    """
    """

    def __init__(self,
                 num_nodes,
                 num_dummies,
                 learnable=True,
                 clamp_at=5.):
        super(AdjEmb, self).__init__()
        self.clamp_value = clamp_at
        self.logits = nn.Parameter(torch.rand(num_nodes, num_nodes + num_dummies) - 0.5, requires_grad=learnable)

    def soft_clip(self, logits):
        return self.clamp_value * torch.tanh(logits / self.clamp_value)

    def forward(self):
        """"""
        return self.soft_clip(self.logits)


class DifferentiableKnnGraphLayer(nn.Module):
    def __init__(self,
                 n_nodes: int,
                 k: int,
                 tau: float,
                 sparsify_gradient=False,
                 at_most_k=False,
                 gradient_sparsity=0.9):
        super(DifferentiableKnnGraphLayer, self).__init__()
        self.k = k
        self.tau = tau
        self.sampler = StraightThroughSubsetSampler(self.k, self.tau)

        self.n_dummies = self.k - 1 if at_most_k else 0
        self.n_nodes = n_nodes
        self.at_most_k = at_most_k

        # self.logits = NodeEmbedding(n_nodes=n_nodes,
        #                             emb_size=n_nodes + self.n_dummies,)
        self.logits = AdjEmb(num_nodes=self.n_nodes, num_dummies=self.n_dummies)
        self.sparsify_gradient = sparsify_gradient
        self.gradient_sparsity = gradient_sparsity
        self.inference_mode = False

    def adj_to_training_edge_index(self, adj):
        if self.sparsify_gradient:
            # randomly select elements for which we will compute gradients
            # even if edge weight is zero
            mask = torch.rand_like(adj) > self.gradient_sparsity
            adj = torch.where(mask, adj + tsl.epsilon, adj)
            return adj_to_edge_index(adj)
        return adj_to_fc_edge_index(adj)

    def sample_adj(self, n_samples=1):
        scores = self.logits()
        if n_samples > 1:
            scores = repeat(scores, "... -> b ...", b=n_samples)
        adj = self.sampler(scores, inference_mode=self.inference_mode)
        if self.n_dummies > 0:
            adj = adj[..., :-self.n_dummies]
        return adj

    def forward(self, x, emb: Tensor):
        if self.training and not self.inference_mode:
            n_samples = x.size(0) # take a sample for each batch
        else:
            n_samples = 1
        adj = self.sample_adj(n_samples=n_samples)
        if self.training and not self.inference_mode:
           return self.adj_to_training_edge_index(adj)
        return adj_to_edge_index(adj)





