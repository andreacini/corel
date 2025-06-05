from einops.layers.torch import Rearrange
from torch import Tensor, nn

from lib.nn.decoder.base_decoder import BaseDecoder
from tsl.nn.blocks import MLP, MLPDecoder, LinearReadout


class MultiQuantileDecoder(BaseDecoder):
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
                 input_size,
                 hidden_size,
                 output_size,
                 quantiles,
                 horizon,
                 n_layers=1):
        super(MultiQuantileDecoder, self).__init__(
            input_size=input_size,
            output_size=output_size,
            horizon=horizon
        )

        self.quantiles = quantiles
        self.n_qs = len(quantiles)
        self.mlp = MLP(input_size=input_size,
                       hidden_size=hidden_size,
                       n_layers=n_layers)
        self.readout = nn.Sequential(
            LinearReadout(input_size=hidden_size,
                          output_size=self.n_qs * output_size,
                          horizon=horizon),
            Rearrange('b t n (q f) -> q b t n f', q=self.n_qs),
        )

    def decode(self, h, *args, **kwargs) -> Tensor:
        if h.dim() == 4:
            # take last step representation
            h = h[:, -1]
        h = self.mlp(h)
        return self.readout(h)
