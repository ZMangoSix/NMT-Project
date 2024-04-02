from torch import nn


# class Classifier():
#     pass


class Encoder(nn.Module):
    """The encoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()


    def foward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """The decoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()


    def init_sate(self, enc_all_outputs, *args):
        raise NotImplementedError
    

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    
    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]
