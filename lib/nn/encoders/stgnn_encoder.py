from einops import rearrange
from torch import nn, Tensor
import torch
from torch_geometric.nn import GatedGraphConv
from torch_geometric.utils import to_undirected

from lib.nn.encoders.base_encoder import InputEncoder
from lib.nn.layers import GraphAnisoConv
from lib.nn.utils import maybe_cat_emb
from tsl.nn.blocks import RNN, MLPDecoder, LinearReadout
from tsl.nn.layers import DiffConv

from tsl.nn.layers.base import NodeEmbedding
from tsl.nn.models import BaseModel
from tsl.nn.utils import maybe_cat_exog
import torch.nn.functional as F


class STGNNEncoder(InputEncoder):
    def __init__(self,
                 input_size: int,
                 n_instances: int,
                 hidden_size: int,
                 emb_size: int,
                 temporal_layers: int,
                 gnn_layers: int,
                 force_symmetric: bool = False,
                 exog_size: int = 0,
                 conv_type: str = "iso",
                 activation: str = 'elu',
                 cat_emb: bool = True,
                 dropout_emb: float = 0.,):
        super(STGNNEncoder, self).__init__(
            input_size=input_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            n_instances=n_instances,
            emb_size=emb_size,
            cat_emb=cat_emb
        )

        self.temporal_encoder = RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            return_only_last_state=True,
            n_layers=temporal_layers
        )

        if conv_type == "iso":
            self.gnn_layers = nn.ModuleList([
                DiffConv(in_channels=hidden_size,
                         out_channels=hidden_size,
                         add_backward=True,
                         k=1,
                         activation=activation) for _ in range(gnn_layers)
            ])
        elif conv_type == "aniso":
            self.gnn_layers = nn.ModuleList([
                GraphAnisoConv(input_size=hidden_size,
                               output_size=hidden_size,
                               activation=activation) for _ in range(gnn_layers)
            ])

        self.conv_type = conv_type
        self.force_symmetric = force_symmetric
        self.dropout_emb = dropout_emb

    def message_passing(self, x, edge_index, edge_weight=None):
        if self.force_symmetric:
            edge_index, edge_weight = to_undirected(edge_index, edge_weight)
        for layer in self.gnn_layers:
            x = layer(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return x

    def encode(self, x, edge_index, edge_weight, disjoint) -> Tensor:
        b, s, n, _ = x.size()

        x = self.input_encoder(x)

        x = self.temporal_encoder(x)

        if disjoint:
            # process the input graphs as single large one
            x = rearrange(x, 'b n f -> (b n) f')

        out = self.message_passing(x,
                                   edge_index=edge_index,
                                   edge_weight=edge_weight)

        out = torch.cat([out, x], dim=-1)

        if disjoint:
            # back to the original shape
            out = rearrange(out, '(b n) f -> b n f', b=b, n=n)

        return out

    @property
    def output_size(self) -> int:
        output_size = super(STGNNEncoder, self).output_size
        output_size += self.hidden_size
        return output_size

    def forward(self, x, edge_index, edge_weight=None, u=None, disjoint=False, **kwargs):
        """"""
        if self.emb is not None:
            emb = F.dropout(self.emb(), self.dropout_emb, self.training)
        else:
            emb = None

        x = maybe_cat_exog(x, u)
        x = maybe_cat_emb(x, emb)

        x = self.encode(x, edge_index, edge_weight, disjoint)

        if self.cat_emb:
            x = maybe_cat_emb(x, emb)

        return x
