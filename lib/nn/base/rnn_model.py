from typing import Optional

from einops import rearrange
from torch import Tensor, nn
from tsl.nn import maybe_cat_exog

from tsl.nn.blocks.decoders import LinearReadout
from tsl.nn.blocks.encoders import RNN, ConditionalBlock
from tsl.nn.models.base_model import BaseModel


class RNNModel(BaseModel):
    r"""Simple RNN for multistep forecasting.

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

    return_type = Tensor

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 cell_type: str = 'gru'):
        super(RNNModel, self).__init__()

        self.input_encoder = nn.Linear(
            input_size + exog_size,
            hidden_size,
        )

        self.rnn = RNN(input_size=hidden_size,
                       hidden_size=hidden_size,
                       n_layers=n_layers,
                       return_only_last_state=True,
                       dropout=dropout,
                       cell=cell_type)

        self.readout = LinearReadout(
            input_size=hidden_size,
            output_size=output_size,
            horizon=horizon
        )

    def forward(self, x: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """"""
        x = maybe_cat_exog(x, u)
        x = self.input_encoder(x)

        x = self.rnn(x)

        return self.readout(x)


class FCRNNModel(RNNModel):
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
                 output_size: int,
                 horizon: int,
                 n_nodes: int,
                 exog_size: Optional[int] = None,
                 hidden_size: int = 32,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 cell_type: str = 'gru'):
        super(FCRNNModel, self).__init__(input_size=input_size * n_nodes,
                                         output_size=output_size * n_nodes,
                                         horizon=horizon,
                                         exog_size=exog_size,
                                         hidden_size=hidden_size,
                                         n_layers=n_layers,
                                         dropout=dropout,
                                         cell_type=cell_type)

    def forward(self, x: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """"""
        # x: [batches, steps, nodes, features]
        # u: [batches, steps, (nodes), features]
        b, _, n, _ = x.size()
        x = rearrange(x, 'b t n f -> b t 1 (n f)')
        if u is not None and u.dim() == 4:
            u = rearrange(u, 'b t n f -> b t 1 (n f)')
        x = super(FCRNNModel, self).forward(x, u)
        # [b, h, 1, (n f)]
        return rearrange(x, 'b h 1 (n f) -> b h n f', n=n)
