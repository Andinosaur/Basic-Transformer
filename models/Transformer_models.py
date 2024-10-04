import copy
import math
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
        return self.decoder(self.encoder(src, src_mask), tgt, src_mask, tgt_mask)
    
    def encoder(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decoder(self, memory, tgt, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    

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
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderLayer(nn.Module):
    def __init__(self, size, self_atten, feed_words, dropout):
        super(EncodeDecoder, self).__init__()
        self.self_atten = self_atten
        self.feed_words = feed_words
        self.sublayer = clone(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_atten(x, x, x, mask))
        return self.sublayer[1](x, self.feed_words)
    

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layer = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layer:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    

class DecoderLayer(nn.Module):
    def __init__(self, size, self_atten, src_atten, feed_words, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_atten = self_atten
        self.src_atten = src_atten
        self.feed_words = feed_words
        self.sublayer = clone(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_atten(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_atten(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_words)
    

def subsquent_mask(size):
    atten_size = (1, size, size)
    subsquent_mask = torch.triu(torch.ones(atten_size), diagonal=1).type(torch.uint8)
    return subsquent_mask == 0


def attention(query, key, value, mask= None, Dropout= None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask=0, value=-1e9)
    p_attn = scores.softmax(dim=-1)
    if Dropout is not None:
        p_attn = Dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadattention(nn.Module):
    def __init__(self, h, d_model, dropout = 0.1):
        super(MultiHeadattention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear = clone(nn.Linear(d_model, d_model), 4)
        self.atten = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask= None):
        if mask is not None:
            mask = mask.unsqueeze(-1)
        nbatch = query.size(0)
        query, key, value = [
            lin(x).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linear, (query, key, value))
        ]
        x, self.atten = attention(query, key, value, mask= mask, Dropout=self.dropout)
        x = (x.transpose(1,2).contiguous().view(nbatch, -1, self.h*self.d_k))
        del query
        del key
        del value
        return self.linear[-1](x)
    

class PostionwiseFeedward(nn.Module):
    def __init__(self, d_model, d_ff, dropout= 0.1):
        super(PostionwiseFeedward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p= dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    

class PostionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        super(PostionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p= dropout)

        pe = torch.zeros(max_len, d_model)
        postion = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(1000.0) / d_model))
        pe[:, 0::2] = torch.sin(postion * div_term)
        pe[:, 1::2] = torch.cos(postion * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)