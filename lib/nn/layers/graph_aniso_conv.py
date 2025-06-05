import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj

from tsl.nn.utils import get_layer_activation


class GraphAnisoConv(MessagePassing):
    r"""Gate Graph Neural Network layer (with residual connections) inspired by
    the FC-GNN model from the paper `"Multivariate Time Series Forecasting with
    Latent Graph Inference" <https://arxiv.org/abs/2203.03423>`_ (Satorras et
    al., 2022).

    Args:
        input_size (int): Input channels.
        output_size (int): Output channels.
        activation (str, optional): Activation function.
        parametrized_skip_conn (bool, optional): If :obj:`True`, then add a
            linear layer in the residual connection even if input and output
            dimensions match.
            (default: :obj:`False`)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: str = 'silu',
                 parametrized_skip_conn: bool = False):
        super(GraphAnisoConv, self).__init__(aggr="add", node_dim=-2)

        self.in_channels = input_size
        self.out_channels = output_size

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * input_size, output_size // 2),
            get_layer_activation(activation)(),
            nn.Linear(output_size // 2, output_size),
            get_layer_activation(activation)(),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(input_size + output_size, output_size),
            get_layer_activation(activation)(),
            nn.Linear(output_size, output_size))

        if (input_size != output_size) or parametrized_skip_conn:
            self.skip_conn = nn.Linear(input_size, output_size)
        else:
            self.skip_conn = nn.Identity()

    def forward(self, x: Tensor, edge_index, edge_weight=None):
        """"""
        if edge_weight is None:
            edge_weight = torch.ones_like(edge_index[0])
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        out = self.update_mlp(torch.cat([out, x], -1)) + self.skip_conn(x)

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight):
        """"""
        mij = self.msg_mlp(torch.cat([x_i, x_j], -1))
        return edge_weight.view(-1, 1) * mij
