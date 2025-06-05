from torch import Tensor, nn

from tsl.utils import foo_signature


class BaseDecoder(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 horizon):

        super(BaseDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.horizon = horizon



    def decode(self, h, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.decode(x, *args, **kwargs)

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
