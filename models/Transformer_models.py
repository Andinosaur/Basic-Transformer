from torch import log_softmax
import torch.nn as nn

class EncodeDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncodeDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decoder(self.encoder(src, src_mask), self.tgt_embed(tgt), src_mask, tgt_mask)
    
    def encoder(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decoder(self, memory, tgt, src_mask, tgt_mask):
        return self.decoder(memory, tgt, src_mask, tgt_mask)
    
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(nn.linear(x), dim=-1)