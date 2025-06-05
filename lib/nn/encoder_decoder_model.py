from tsl.nn.models import BaseModel


class EncoderDecoderModel(BaseModel):
    return_type = None
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 encoder_class,
                 decoder_class,
                 encoder_kwargs = None,
                 decoder_kwargs = None,
                 exog_size: int = 0,):
        super(EncoderDecoderModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.horizon = horizon
        encoder_kwargs = encoder_kwargs or dict()

        self.encoder = encoder_class(input_size=input_size,
                                     exog_size=exog_size,
                                     **encoder_kwargs)
        self.decoder = decoder_class(input_size=self.encoder.output_size,
                                     output_size=output_size,
                                     horizon=horizon,
                                     **decoder_kwargs)

    def forward(self, *args, decoder_kwargs=None, **kwargs):
        out = self.encoder(*args, **kwargs)
        decoder_kwargs = decoder_kwargs or dict()
        return self.decoder(out, **decoder_kwargs)
