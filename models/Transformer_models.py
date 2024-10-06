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


def clones(Modules, N):
    return nn.ModuleList([copy.deepcopy(Modules) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, feature, eps = 1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.a_w = torch.ones(feature)
        self.b_w = torch.zeros(feature)
        self.eps = eps

    def forward(self, x):
        mean = x.means(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_w * (x - mean) / (std + self.eps) + self.b_w


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1) -> None:
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


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

    def forward(self, x, m, src_mask, tgt_mask):
        for layer in self.layer:
            x = layer(x, m, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab) -> None:
        super(Generator, self).__init__()
        self.d_model = d_model
        self.norm = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_atten, Feedwards, dropout=0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.self_atten = self_atten
        self.Feedwards = Feedwards
        self.dropout = nn.Dropout(p=dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_atten(x, x, x, mask))
        x = self.sublayer[1](x, self.Feedwards)
        return self.dropout(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, src_atten, self_atten, Feedwards, dropout) -> None:
        super(DecoderLayer, self).__init__()
        self.src_atten = src_atten
        self.self_atten = self_atten
        self.Feedwards = Feedwards
        self.dropout = nn.Dropout(p=dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, m, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.src_atten(x, m, m, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.self_atten(x, x, x, src_mask))
        x = self.sublayer[2](x, self.Feedwards)
        return self.dropout(x)


def subsequent_mask(size):
    subsequent_mask = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(subsequent_mask), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill_(mask==0, -1e9)
    p_ttn = scores.softmax(dim=-1)
    if dropout is not None:
        p_ttn = dropout(p_ttn)
    return torch.matmul(p_ttn, value), p_ttn


class MultiHeadattention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1) -> None:
        super(MultiHeadattention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.dropout = nn.Dropout(p=dropout)
        self.linear = clones(nn.Linear(d_model, d_model), 4)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        nbatch = query.size(0)
        query, key, value = [
            lin(x).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linear, (query, key, value))
        ]
        x, p_ttn = attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.h * self.d_k)
        return self.linear[-1](x), p_ttn


class Embeddings(nn.Module):
    def __init__(self, d_model, vacob) -> None:
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.Embed = nn.Embedding(vacob, d_model)

    def forward(self, x):
        return self.Embed(x) * math.sqrt(self.d_model)


class PostionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1) -> None:
        super(PostionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        postion = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(1000.0) / d_model)
        pe[:, 0::2] = torch.sin(postion * div_term)
        pe[:, 1::2] = torch.cos(postion * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class PostionwiseFeedwards(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1) -> None:
        super(PostionwiseFeedwards, self).__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.lin2(self.lin1(x).relu()))


def make_model(src_vacob, tgt_vacob, N=6, h=8, d_model=512, d_ff=2048, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadattention(d_model, h, dropout)
    ff = PostionwiseFeedwards(d_model, d_ff, dropout)
    postion = PostionalEncoding(d_model=d_model, dropout=dropout)
    model = EncodeDecode(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vacob), c(postion)),
        nn.Sequential(Embeddings(d_model, tgt_vacob), c(postion)),
        Generator(d_model, tgt_vacob)
    )
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return model