from einops import repeat
from typing import Optional

from einops import rearrange
from torch import Tensor, nn

from lib.nn.encoders.base_encoder import InputEncoder
from tsl.nn.blocks import MultiRNN

from tsl.nn.blocks.encoders import RNN


class RNNEncoder(InputEncoder):
    def __init__(self,
                 input_size: int,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 cell_type: str = 'gru',
                 emb_size: int = 0,
                 n_instances: Optional[int] = None,
                 cat_emb=False
                 ):
        super(RNNEncoder, self).__init__(
            input_size=input_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            emb_size=emb_size,
            n_instances=n_instances,
            cat_emb=cat_emb
        )

        self.rnn = RNN(input_size=hidden_size,
                       hidden_size=hidden_size,
                       n_layers=n_layers,
                       return_only_last_state=True,
                       dropout=dropout,
                       cell=cell_type)

    def encode(self, x) -> Tensor:
        """"""
        x = super(RNNEncoder, self).encode(x)

        x = self.rnn(x)

        return x


class LocalRNNEncoder(InputEncoder):
    def __init__(self,
                 input_size: int,
                 n_instances: int,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 cell_type: str = 'gru'):
        super(LocalRNNEncoder, self).__init__(
            input_size=input_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            emb_size=0,
            n_instances=n_instances,
            cat_emb=False
        )

        self.rnns = MultiRNN(input_size=hidden_size,
                             n_instances=n_instances,
                             hidden_size=hidden_size,
                             n_layers=n_layers,
                             return_only_last_state=True,
                             dropout=dropout,
                             cell=cell_type)

    def encode(self, x) -> Tensor:
        super(LocalRNNEncoder, self).encode(x)
        return self.rnns(x)


class FCRNNEncoder(RNNEncoder):
    r"""A simple fully connected RNN for multistep forecasting that simply
    flattens data along the spatial dimension.

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of units in the recurrent cell.
        output_size (int): Number of output channels.
        ff_size (int): Number of units in the link predictor.
        exog_size (int): Size of the exogenous variables.
        n_layers (int): Number of RNN layers.
        ff_layers (int): Number of hidden layers in the decoder.
        dropout (float, optional): Dropout probability in the RNN encoder.
        ff_dropout (float, optional): Dropout probability int the GCN decoder.
        horizon (int): Forecasting horizon.
        cell_type (str, optional): Type of cell that should be use
            (options: [:obj:`gru`, :obj:`lstm`]).
            (default: :obj:`gru`)
        activation (str, optional): Activation function.
            (default: :obj:`relu`)
    """

    def __init__(self,
                 input_size: int,
                 n_instances: int,
                 exog_size: Optional[int] = None,
                 hidden_size: int = 32,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 cell_type: str = 'gru'):
        super(FCRNNEncoder, self).__init__(input_size=input_size * n_instances,
                                           exog_size=exog_size,
                                           hidden_size=hidden_size,
                                           n_layers=n_layers,
                                           dropout=dropout,
                                           cell_type=cell_type,
                                           emb_size=0,
                                           n_instances=n_instances,
                                           cat_emb=False)

    def forward(self, x: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """"""
        # x: [batches, steps, nodes, features]
        # u: [batches, steps, (nodes), features]
        b, _, n, _ = x.size()
        x = rearrange(x, 'b t n f -> b t 1 (n f)')
        if u is not None and u.dim() == 4:
            u = rearrange(u, 'b t n f -> b t 1 (n f)')
        x = super(FCRNNEncoder, self).forward(x, u)
        # [b, h, 1, (n f)]
        return repeat(x, 'b h 1 f -> b h n f', n=n)
