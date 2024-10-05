import copy
import math
from torch import log_softmax
import torch
import torch.nn as nn

class EncodeDecode(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator) -> None:
        super(EncodeDecode, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decoder(tgt, self.encoder(src, src_mask), src_mask, tgt_mask)

    def encoder(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decoder(self, tgt, memory, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Encoder(nn.Module):
    def __init__(self, layer, N) -> None:
        super(Encoder, self).__init__()
        self.layer = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layer:
            x = layer(x, mask)
        return self.norm(x)

    
class Decoder(nn.Module):
    def __init__(self, layer, N) -> None:
        super(Decoder, self).__init__()
        self.layer = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layer:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Generator(nn.Module):
    def __init__(self, d_model, vocab) -> None:
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_atten, feedward, dropout=0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_atten = self_atten
        self.feedward = feedward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_atten(x, x, x, mask))
        return self.sublayer[1](x, self.feedward)
    

class DecoderLayer(nn.Module):
    def __init__(self, size, src_atten, self_atten, feedward, dropout=0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.size = size
        self.src_atten = src_atten
        self.self_atten = self_atten
        self.feedward = feedward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, m, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.src_atten(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.self_atten(x, m, m, src_mask))
        return self.sublayer[2](x, self.feedward)
    

def clones(Modules, N):
    nn.ModuleList(copy.deepcopy(Modules) for _ in range(N))


class LayerNorm(nn.Module):
    def __init__(self, feature, eps=1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.a_w = torch.ones(feature)
        self.b_w = torch.zeros(feature)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_w * (x - mean) / (std + self.eps) + self.b_w
    

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout) -> None:
        super(SublayerConnection, self).__init__()
        self.size = size
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

def attention(query, key, value, mask = None, dropout = None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill_(mask==0, -1e9)
    p_ttn = scores.softmax(dim=-1)
    if dropout is not None:
        p_ttn = dropout(p_ttn)
    return torch.matmul(p_ttn, value), p_ttn


def subsquent_mask(size):
    atten_shape = (1, size, size)
    mask = torch.triu(torch.ones(atten_shape), diagonal=1).type(torch.uint8)
    return mask == 0


class MultiHeadattenion(nn.Module):
    def __init__(self, h, d_model, dropout=0.1) -> None:
        super(MultiHeadattenion, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.atten = None
        self.dropout = nn.Dropout(p=dropout)
        self.Linear = clones(nn.Linear(d_model, d_model), 4)

    def forward(self, query, key, value, mask= None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatch = query.size(0)
        query, key, value = [
            lin(x).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.Linear, (query, key, value))
        ]
        x, self.atten = attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.h*self.d_k)
        del query
        del key
        del value
        return self.Linear[-1](x)
        

class PostionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1) -> None:
        super(PostionwiseFeedForward, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.Lin1 = nn.Linear(d_model, d_ff)
        self.Lin2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.Lin2(self.dropout(self.Lin1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab) -> None:
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositonalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000) -> None:
        super(PositonalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        postion = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(1000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(postion * div_term)
        pe[:, 1::2] = torch.cos(postion * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadattenion(h, d_model, dropout)
    ff = PostionwiseFeedForward(d_model, d_ff, dropout)
    postion = PositonalEncoding(d_model, dropout)
    model = EncodeDecode(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(postion)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(postion)),
        Generator(d_model, tgt_vocab)
    )
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return model
