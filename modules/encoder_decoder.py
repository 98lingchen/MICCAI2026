from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .att_model import pack_wrapper, AttModel


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0



class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm

    def forward(self, src, tgt, src_mask, tgt_mask, lambd=1.0):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, lambd=lambd)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask, lambd=1.0):
        memory_init = self.rm.init_memory(hidden_states.size(0)).to(hidden_states.device)
        memory = self.rm(self.tgt_embed(tgt), memory_init)
        

        memory_last = memory[:, -1:, :]
        
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory_last, lambd=lambd)



class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory, lambd=1.0):
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory, lambd=lambd)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory, lambd=1.0):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory, lambd=lambd)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory, lambd=lambd)
        return self.sublayer[2](x, self.feed_forward, memory, lambd=lambd)

class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory, lambd=1.0):
        return x + self.dropout(sublayer(self.norm(x, memory, lambd=lambd)))





class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))

        self.rm_dim = rm_num_slots * rm_d_model

        self.mlp_gamma = nn.Sequential(
            nn.Linear(self.rm_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.mlp_beta = nn.Sequential(
            nn.Linear(self.rm_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )

        self.lambda_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh()
        )

    def forward(self, x, memory, lambd):

        B, T, D = x.shape


        if memory.size(0) != B:
            raise RuntimeError(f"[ConditionalLN] memory B={memory.size(0)} != x B={B}, "
                               f"memory shape={tuple(memory.shape)}, x shape={tuple(x.shape)}")


        memory = memory.reshape(B, -1)  # [B, rm_dim]
        if memory.size(1) != self.rm_dim:
            raise RuntimeError(f"[ConditionalLN] memory flatten dim={memory.size(1)} != rm_dim={self.rm_dim}, "
                               f"orig memory shape={tuple(memory.shape)}")

        delta_gamma = self.mlp_gamma(memory)  # [B,D]
        delta_beta  = self.mlp_beta(memory)   # [B,D]


        if not torch.is_tensor(lambd):
            lambd = torch.tensor(lambd, device=x.device, dtype=x.dtype)
        lambd = lambd.to(device=x.device, dtype=x.dtype)

        if lambd.dim() == 0:
            lambd = lambd.view(1, 1).expand(B, 1)
        elif lambd.dim() == 1:
            if lambd.numel() != B:
                raise RuntimeError(f"[ConditionalLN] lambd numel={lambd.numel()} != B={B}, shape={tuple(lambd.shape)}")
            lambd = lambd.view(B, 1)
        elif lambd.dim() == 2 and lambd.size(0) == B and lambd.size(1) == 1:
            pass
        else:
            raise RuntimeError(f"[ConditionalLN] bad lambd shape={tuple(lambd.shape)} for B={B}")

        lambda_emb = self.lambda_proj(lambd)  # [B,D]


        delta_gamma = (delta_gamma + lambda_emb).unsqueeze(1)  # [B,1,D]
        delta_beta  = (delta_beta  + lambda_emb).unsqueeze(1)

        gamma_hat = self.gamma.view(1, 1, -1) + delta_gamma
        beta_hat  = self.beta.view(1, 1, -1) + delta_beta


        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (std + self.eps)

        return gamma_hat * x_norm + beta_hat




class RelationalMemory(nn.Module):
    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        self.num_slots, self.num_heads, self.d_model = num_slots, num_heads, d_model
        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU())
        self.W = nn.Linear(d_model, d_model * 2)
        self.U = nn.Linear(d_model, d_model * 2)

    def init_memory(self, batch_size):
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            pad = torch.zeros((batch_size, self.num_slots, self.d_model - self.num_slots))
            memory = torch.cat([memory, pad], -1)
        else:
            memory = memory[:, :, :self.d_model]
        return memory

    def forward_step(self, input, memory):
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = v = torch.cat([memory, input.unsqueeze(1)], 1)
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)
        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        input_gate, forget_gate = torch.split(gates, self.d_model, dim=2)
        next_memory = torch.sigmoid(input_gate) * torch.tanh(next_memory) + torch.sigmoid(forget_gate) * memory
        return next_memory.reshape(-1, self.num_slots * self.d_model)

    def forward(self, inputs, memory):
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        return torch.stack(outputs, dim=1)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None: mask = mask.unsqueeze(1)
        nb = query.size(0)
        query, key, value = [l(x).view(nb, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nb, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1, self.w_2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x): return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1)])



class EncoderDecoder(AttModel):
    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        return Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model), self.num_layers),
            lambda x: x, nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)), rm)

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers, self.d_model, self.d_ff = args.num_layers, args.d_model, args.d_ff
        self.num_heads, self.dropout = args.num_heads, args.dropout
        self.rm_num_slots, self.rm_num_heads, self.rm_d_model = args.rm_num_slots, args.rm_num_heads, args.rm_d_model
        self.model = self.make_model(self.vocab_size + 1)
        self.logit = nn.Linear(args.d_model, self.vocab_size + 1)

    def init_hidden(self, bsz): return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, _, att_masks, _ = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)
        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        if att_masks is None: att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)
        seq_mask = None
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True
            seq_mask = seq_mask.unsqueeze(-2) & subsequent_mask(seq.size(-1)).to(seq.device)
        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, lambd=1.0):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out = self.model(att_feats, seq, att_masks, seq_mask, lambd=lambd)
        return F.log_softmax(self.logit(out), dim=-1)


    def _sample(self, fc_feats, att_feats, att_masks=None, update_opts={}, lambd=1.0):
        opt = self.args.__dict__.copy()
        opt.update(update_opts)
        

        sample_n = opt.get('sample_n', 1)
        sample_method = opt.get('sample_method', 'greedy')
        

        if sample_n > 1:

            def expand_tensor(t):
                if t is None: return None

                return t.repeat_interleave(sample_n, dim=0)

            fc_feats = expand_tensor(fc_feats)
            att_feats = expand_tensor(att_feats)
            att_masks = expand_tensor(att_masks)
            

            if isinstance(lambd, torch.Tensor):
                lambd = expand_tensor(lambd)
            elif isinstance(lambd, (float, int)):

                pass

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        

        p_fc, p_att, memory, p_mask = self._prepare_feature(fc_feats, att_feats, att_masks)
        
        seq = fc_feats.new_full((batch_size, self.max_seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.max_seq_length, self.vocab_size + 1)
        

        for t in range(self.max_seq_length + 1):
            if t == 0: 
                it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
            
            logprobs, state = self.get_logprobs_state(it, p_fc, p_att, memory, p_mask, state, lambd=lambd)
            
            if t == self.max_seq_length: break
            

            it, _ = self.sample_next_word(logprobs, sample_method, 1.0)
            
            if t == 0: 
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx
                logprobs = logprobs * unfinished.unsqueeze(1).float()
                unfinished = unfinished * (it != self.eos_idx)
            
            seq[:, t], seqLogprobs[:, t] = it, logprobs
            
            if unfinished.sum() == 0: break
            

        return seq, seqLogprobs

    def get_logprobs_state(self, it, fc_feats, att_feats, memory, att_masks, state, lambd=1.0):
        xt = self.embed(it)
        output, state = self.core(xt, fc_feats, att_feats, memory, state, att_masks, lambd=lambd)
        return F.log_softmax(self.logit(output), dim=1), state

    def core(self, it, fc_ph, att_ph, memory, state, mask, lambd=1.0):
        if len(state) == 0: ys = it.unsqueeze(1)
        else: ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(ys.device), lambd=lambd)
        return out[:, -1], [ys.unsqueeze(0)]