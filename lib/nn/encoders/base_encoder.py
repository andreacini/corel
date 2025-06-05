from typing import Optional

from torch import Tensor, nn

from lib.nn.utils import maybe_cat_emb
from tsl.nn import maybe_cat_exog
from tsl.nn.layers import NodeEmbedding
from tsl.utils import foo_signature


class InputEncoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 exog_size: int = 0,
                 emb_size: int = 0,
                 n_instances: Optional[int] = None,
                 cat_emb=False):

        super(InputEncoder, self).__init__()

        if emb_size > 0:
            assert n_instances is not None
            self.emb = NodeEmbedding(n_nodes=n_instances, emb_size=emb_size)
        else:
            self.register_parameter('emb', None)

        self.cat_emb = cat_emb
        self.hidden_size = hidden_size
        self.emb_size = emb_size

        self.input_encoder = nn.Linear(
            input_size + exog_size + emb_size,
            hidden_size,
        )

    def encode(self, x) -> Tensor:
        return self.input_encoder(x)

    def forward(self, x: Tensor, u: Optional[Tensor] = None, *args, **kwargs) -> Tensor:
        """"""
        if self.emb is not None:
            emb = self.emb()
        else:
            emb = None

        x = maybe_cat_exog(x, u)
        x = maybe_cat_emb(x, emb)

        x = self.encode(x)

        if self.cat_emb:
            x = maybe_cat_emb(x, emb)

        return x

    @property
    def output_size(self) -> int:
        output_size = self.hidden_size
        if self.cat_emb:
            output_size += self.emb_size
        return output_size


    @classmethod
    def get_signature(cls) -> dict:
        """Get signature of the model's
        :class:`~tsl.nn.models.BaseModel`'s :obj:`__init__` function."""
        return foo_signature(cls)

    @classmethod
    def filter_init_args_(cls, mapping: dict):
        """Remove from :attr:`mapping` all the keys that are not in
        :class:`~tsl.nn.models.BaseModel`'s :obj:`__init__` function."""
        model_sign = cls.get_signature()
        if model_sign['has_kwargs']:
            return
        model_signature = model_sign['signature']
        del_keys = filter(lambda k: k not in model_signature, mapping.keys())
        for k in list(del_keys):
            del mapping[k]
