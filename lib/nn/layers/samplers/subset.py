import torch
from torch import nn

import tsl

def k_hot_topk(scores, k):
    khot = torch.zeros_like(scores)
    _, ind = torch.topk(scores, k, dim=-1)
    khot.scatter_(-1, ind, 1.)
    return khot

def relaxed_gumbel_top_k(scores, k, tau):
    # sample a gumbel for each node
    g = - torch.log(-torch.log(torch.rand_like(scores)))
    scores = scores + g

    # continuous top k
    relaxed_khot = torch.zeros_like(scores)
    onehot_approx = torch.zeros_like(scores)
    for i in range(k):
        khot_mask = torch.clip(1.0 - onehot_approx, min=tsl.epsilon)
        scores = scores + torch.log(khot_mask)
        onehot_approx = torch.nn.functional.softmax(scores / tau, dim=-1)
        relaxed_khot = relaxed_khot + onehot_approx
    return relaxed_khot


class StraightThroughSubsetSampler(nn.Module):
    """"""
    def __init__(self, k, tau):
        super(StraightThroughSubsetSampler, self).__init__()
        self.k = k
        self.tau = tau

    def forward(self, scores, inference_mode=False):
        if self.training and not inference_mode:
            sample = relaxed_gumbel_top_k(scores=scores,
                                          k=self.k,
                                          tau=self.tau)
            # top k
            khot = k_hot_topk(sample, self.k)
            return (khot - sample).detach() + sample
        return k_hot_topk(scores, self.k)