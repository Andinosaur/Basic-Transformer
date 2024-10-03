import copy
from torch import log_softmax
import torch
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
    

def clone(models, N):
    return nn.ModuleList([copy.deepcopy(models)for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layer = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        x = self.layer(x, mask)
        return self.norm(x)
    

class LayerNorm(nn.Module):
    def __init__(self, feature, eps= 1e-6):
        super(LayerNorm, self).__init__()
        self.a_c = nn.parameter(torch.ones(feature))
        self.b_c = nn.parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim= True)
        std = x.std(-1, keepdim= True)
        return self.a_c * (x - mean) / (std + self.eps) + self.b_c


class SublayerConnection(nn.Module):
    def __init__(self, size):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        
    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))
    

class EncoderLayer(nn.Module):
    def __init__(self, atten_layer, feed_words, size):
        super(EncoderLayer, self).__init__()
        self.atten_layer = atten_layer
        self.feed_words = feed_words
        self.norm = clone(SublayerConnection(size), 2)

    def forward(self, x):
        x = self.norm[0](self.atten_layer(x))
        return self.norm[1](self.feed_words(x))