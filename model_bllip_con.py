import copy
import heapq
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from masking_bllip import utils as masking_utils
# from masking_bllip import masking_types as types
import time
# from helping_utils.logger import configure_logger, get_logger
from dataclasses import dataclass
from typing import Optional, Callable, List, Union, Tuple
import logging
# logger = get_logger()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class PositionalEmbedding(nn.Module): # also posemb for pushdown
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:] # r * None * d_model

class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)
        # self.layer_norm = nn.Identity()
        self.pre_lnorm = pre_lnorm
        
    # def init_weights(self):
    #     nn.init.xavier_uniform_(self.CoreNet[0].weight)
    #     nn.init.xavier_uniform_(self.CoreNet[3].weight)

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output

class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head # n_heads
        self.d_model = d_model # state_size
        self.d_head = d_head # projection_size = state_size // n_heads
        self.dropout = dropout

        self.qkv_net = nn.Sequential(
            nn.Linear(d_model, 3 * n_head * d_head, bias=False),
            nn.Dropout(dropout)
        )

        self.drop = nn.Dropout(dropout) # dropout
        self.dropatt = nn.Dropout(dropout) # dropout
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False) # multi_head_merge

        self.layer_norm = nn.LayerNorm(d_model)
        # self.layer_norm = nn.Identity()
        self.scale = 1 / (d_head ** 0.5) # scale

        self.pre_lnorm = pre_lnorm
        
    # def init_weights(self):
    #     nn.init.normal_(self.qkv_net[0].weight, 0.0, 0.02)
    #     nn.init.normal_(self.o_net.weight, 0.0, 0.02)
    
    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        # self.depth_embed = torch.nn.Embedding(151, self.n_head * self.d_head)

    # def init_weights(self):
    #     nn.init.normal_(self.r_net.weight, 0.0, 0.02)
    #     super().init_weights()
        
    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, past_keys=None, past_values=None, cache=False):
        if cache or past_keys or past_values or mems is not None:
            raise NotImplementedError("We ignore TXL mems for now.")
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        if cache:
            new_key = w_head_k.view(qlen, bsz, -1)
            new_value = w_head_v.view(qlen, bsz, -1)
        
        if past_keys is not None:
            w_head_k = torch.cat([past_keys, w_head_k], dim=0)
            w_head_v = torch.cat([past_values, w_head_v], dim=0)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.reshape(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.dropatt(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        if cache:
            return output, new_key, new_value
        else:
            return output

class PushdownMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, pre_lnorm=False, max_stack_depth=150):
        super(PushdownMultiHeadAttn, self).__init__()
        self.n_head = n_head # n_heads | n_head
        self.d_model = d_model # state_size | n_embd
        self.d_head = d_head # projection_size = state_size // n_heads | head_dim
        # self.dropout = dropout 
        self.dropout = nn.Dropout(dropout) # for ordinary & attn
        
        # self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False) # c_attn
        self.qkv_net = nn.Sequential(
            nn.Linear(d_model, 3 * n_head * d_head, bias=False),
            nn.Dropout(dropout)
        )
        
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False) # multi_head_merge | c_proj, n_head * d_head -> d_model
        # we expect n_head * d_head == d_model, but if not, we can also handle it
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / (d_head ** 0.5)
        
        self.pre_lnorm = pre_lnorm
        self.max_stack_depth = max_stack_depth
        
        self.beta = nn.Embedding(max_stack_depth, self.n_head * self.d_head) # depth_embed
        
        self.r_net = nn.Linear(d_model, n_head * d_head, bias=False) # r_net for projecting sinuoidal positional embedding

    # def init_weights(self):
    #     nn.init.normal_(self.qkv_net[0].weight, 0.0, 0.02)
    #     nn.init.normal_(self.o_net.weight, 0.0, 0.02)
    #     nn.init.normal_(self.r_net.weight, 0.0, 0.02)
        
    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, stack_tape, attn_mask=None, mems=None, past_keys=None, past_values=None, cache=False):
        # w: input, r: posemb
        # w shape: [qlen, bsz, d_model]
        # r shape: [rlen, d_model] <-- sinuoidal positional embedding, used for relative R_{i-j}
        # r_w_bias shape: [n_head, d_head] <-- learnable term u for term C 
        # r_r_bias shape: [n_head, d_head] <-- learnable term v for term D
        # stack_tape shape: [bsz, klen, klen] <-- stack depth information
        # attn_mask shape: [qlen, klen] or [qlen, klen, bsz] <-- mask for attention score
        
        
        # ref: pushdown layers impl
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        # B: batch size | bsz
        # T: sequence length | qlen (decoded sequence length)
        # T': mems length | klen (we assume this to be stack_tape length)
        # C: state size | d_model
        
        if mems or past_keys or past_values or cache:
            raise NotImplementedError("We ignore TXL mems for now.")
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r) # shape [rlen, n_head * d_head] <-- This is projected R_{i-j} for term B/D

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1) # k/q/v shape [klen=qlen, bsz, n_head, d_head]


        if cache:
            new_key = w_head_k.view(qlen, bsz, -1)
            new_value = w_head_v.view(qlen, bsz, -1)
        
        if past_keys is not None:
            w_head_k = torch.cat([past_keys, w_head_k], dim=0)
            w_head_v = torch.cat([past_values, w_head_v], dim=0)

        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        # shape of w_head_k: [klen, bsz, n_head, d_head] = [T', B, n_head, head_dim]
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        assert klen == qlen, "We expect klen == qlen, but got klen = %d, qlen = %d" % (klen, qlen)
        

        # NOTE: ADD STACK EMB (DEPTH EMB) HERE to w_head_k
        # 通过堆栈 tape 得到深度嵌入：
        # stack_tape: (B, T') -> beta 映射后为 (B, T', head_dim)
        # unsqueeze 后变为 (B, 1, T', head_dim)，便于广播到每个 head
        

        rr_head_q = w_head_q + r_r_bias
        # NOTE: Original ABCD calculation in TXL
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # shape [qlen, klen, bsz, n_head] = [T, T', B, n_head]
        BD = self._rel_shift(BD)
        
        attn_score = None
        if True: # DONE: gather to save gpu memory
            # NOTE: We want to make attn_score = AC_0 (biased q * ordinary k) + AC_depth (biased q * depth component for k) + BD 
            
            # 1. Compute AC_0
            # AC w/o beta: AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k)) # shape [qlen, klen, bsz, n_head] = [T, T', B, n_head]
            rw_head_q = w_head_q + r_w_bias  # shape [qlen, bsz, n_head, d_head] + [n_head, d_head] = [qlen, bsz, n_head, d_head]
            # w_head_k shape: [klen, bsz, n_head, d_head] = [T', B, n_head, head_dim]
            # AC: rw_head_q @ w_head_k at last dim
            
            # AC_0 but we can in place
            AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k)) # shape [qlen, klen, bsz, n_head] = [T, T', B, n_head]
            
            # 2. Compute AC_depth
            # 2.1 Compute the table from which AC_depth gathers
            
            # d_emb_table: shape [bsz, max_stack_depth, n_head*d_head]
            stack_tape = stack_tape.clamp(0, self.max_stack_depth - 1) # stack_tape: [bsz, qlen, klen] = [B, T, T']
            d_emb_table = self.beta(torch.arange(self.max_stack_depth, device=stack_tape.device).int()) # shape [max_stack_depth, n_head*d_head]
            
            # Moderate_wk in main branch [max_stack_depth, n_head, d_head] = [max_stack_depth, n_head, head_dim]
            table_wk = d_emb_table.view(self.max_stack_depth, self.n_head, self.d_head)
            
            # Moderate_AC in main branch
            table_AC_depth = torch.einsum('ibnd,jnd->ijbn', (rw_head_q, table_wk)) # shape [qlen, max_stack_depth, bsz, n_head] = [T, max_stack_depth, B, n_head]
            
            # 2.2 Gather the table according to stack_tape & Add to AC_0
            stack_tape = stack_tape.permute(1, 2, 0) # shape [qlen, klen, bsz] = [T, T', B]
            # key pos: j, query pos: k. We use j to gather the table
            # stack_tape.unsqueeze(-1) expands the shape to [qlen, klen, bsz, 1] 
            # -> expand: [qlen, klen, bsz, n_head] 
            # -> ok for gather
            
            # OLD
            # AC_depth_gathered = table_AC_depth.gather(1, stack_tape.unsqueeze(-1).expand(-1, -1, -1, self.n_head))
            # attn_score = (AC + BD) + AC_depth_gathered # shape [qlen, klen, bsz, n_head] = [T, T', B, n_head]
            
            # NEW
            attn_score = AC + BD + table_AC_depth.gather(1, stack_tape.unsqueeze(-1).expand(-1, -1, -1, self.n_head)) # shape [qlen, klen, bsz, n_head] = [T, T', B, n_head]
            
            attn_score.mul_(self.scale)
        else:
            # stack_tape: [bsz, qlen, klen] = [B, T, T'], stack_tape[b][i][j] -> d_{i,j} in batch b
            stack_tape = stack_tape.clamp(0, self.max_stack_depth - 1) # stack_tape: [bsz, qlen, klen] = [B, T, T']
            depth_emb = self.beta(stack_tape.int()) # shape [bsz, qlen, klen, n_head*d_head] = [B, T, T', n_head*head_dim]
            # _, depth_emb, _ = self.qkv_net(depth_emb).chunk(3, dim=-1) # shape [bsz, qlen, klen, n_head*d_head] = [B, T, T', n_head*head_dim]
            # d_{i,j} in batch b is depth_emb[b][i][j]
            # depth_emb = [bsz, qlen, klen, n*d_head] -> [bsz, qlen, klen, n_head, d_head]
            depth_emb = depth_emb.view(bsz, depth_emb.size(1), depth_emb.size(2), self.n_head, self.d_head)
            
            
            # w_head_k shape: [klen, bsz, n_head, d_head] = [T', B, n_head, head_dim]
            # depth_emb shape: [bsz, qlen, klen, n_head, d_head] = [B, T, T', n_head, head_dim]
            # w_head_k should be [bsz, 1, klen, n_head, d_head] to broadcast with depth_emb
            w_head_k = w_head_k.permute(1, 0, 2, 3).unsqueeze(1) # shape [B, 1, T', n_head, head_dim] = [bsz, 1, klen, n_head, d_head]
            w_head_k = w_head_k + depth_emb # shape [B, T, T', n_head, d_head] = [bsz, qlen, klen, n_head, head_dim]
            torch.cuda.empty_cache()

            #### compute attention score
            rw_head_q = w_head_q + r_w_bias  # shape [qlen, bsz, n_head, d_head] + [n_head, d_head] = [qlen, bsz, n_head, d_head]
            # rw_head_q: [qlen, bsz, n_head, d_head] = [T, B, n_head, head_dim]
            # w_head_k: [bsz, qlen, klen, n_head, d_head] = [B, T, T', n_head, head_dim]
            # NOTE: want AC: [qlen, klen, bsz, n_head] = [T, T', B, n_head] as the @ product of rw_head_q and w_head_k
            AC = torch.einsum("ibhd,bijhd->ijbh", rw_head_q, w_head_k)

            # [qlen x klen x bsz x n_head]
            attn_score = AC + BD
            attn_score.mul_(self.scale)
        try:
            if attn_mask is not None and attn_mask.any().item():
                if attn_mask.dim() == 2:
                    attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
                elif attn_mask.dim() == 3:
                    attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))
        except Exception as e:
            # print shapes
            print("attn_score: ", attn_score.shape)
            print("attn_mask: ", attn_mask.shape)
            print("w: ", w.shape)
            print("r: ", r.shape)
            print("r_w_bias: ", r_w_bias.shape)
            print("r_r_bias: ", r_r_bias.shape)
            print("stack_tape: ", stack_tape.shape)
            raise e
            

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        # attn_prob = self.dropatt(attn_prob) # FIXME: MAYBE WE NEED SEPARATE DROPOUT FOR ATTENTION
        attn_prob = self.dropout(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.reshape(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.dropout(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)
        torch.cuda.empty_cache()

        # from utils import tensor_memory
        # print(tensor_memory(output))
        # print("Memory Allocated: ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
        # print("Memory Reserved: ", torch.cuda.memory_reserved() / 1024 / 1024, "MB")
        if cache:
            return output, new_key, new_value
        else:
            return output
        
        
class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropoutf, dropouta,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropouta, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropoutf, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))
        
    # def init_weights(self):
    #     self.dec_attn.init_weights()
    #     self.pos_ff.init_weights()
    
    def forward(self, dec_inp, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, past_keys=None, past_values=None, cache=False):
        if cache:
            output, new_key, new_value = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                                attn_mask=attn_mask, mems=mems, past_keys=past_keys, past_values=past_values, cache=cache)
            output = self.pos_ff(output)

            return output, new_key, new_value
        else:
            output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                                attn_mask=attn_mask, mems=mems, past_keys=past_keys, past_values=past_values, cache=cache)
            output = self.pos_ff(output)
            
            return output


class RelPartialLearnablePushdownLayer(RelPartialLearnableDecoderLayer):
    # the dec_attn is PushdownMultiHeadAttn
    def __init__(self, n_head, d_model, d_head, d_inner, dropoutf, dropouta,
                    **kwargs):
        super(RelPartialLearnablePushdownLayer, self).__init__(n_head, d_model, d_head, d_inner, dropoutf, dropouta, pre_lnorm=kwargs.get('pre_lnorm'))
        self.dec_attn = PushdownMultiHeadAttn(n_head, d_model, d_head, dropouta, 
                                              pre_lnorm=kwargs.get('pre_lnorm'), max_stack_depth=kwargs.get('max_stack_depth'))
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropoutf,
                                     pre_lnorm=kwargs.get('pre_lnorm')) # the original paper seems not adding this??
        
    # def init_weights(self):
    #     self.dec_attn.init_weights()
    #     self.pos_ff.init_weights()
        
    def forward(self, dec_inp, r, r_w_bias, r_r_bias, stack_tape, attn_mask=None, mems=None, past_keys=None, past_values=None, cache=False):
        if cache:
            output, new_key, new_value = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias, stack_tape, attn_mask=attn_mask, mems=mems, past_keys=past_keys, past_values=past_values, cache=cache)
            output = self.pos_ff(output)
            return output, new_key, new_value
        else:
            output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias, stack_tape, attn_mask=attn_mask, mems=mems, past_keys=past_keys, past_values=past_values, cache=cache)
            output = self.pos_ff(output)
            
            return output


class AttachmentHead(nn.Module):
    """
    AttachmentHead 实现了根据当前隐藏状态和下一个预测 token，
    利用注意力机制选择一个候选 constituent（通过右侧 token）进行 reduce，
    参考自论文附录 Figure 8。

    输入：
      - x: (B, T, n_embd) 隐藏状态序列
      - stack_tape: (B, T, T) 的堆栈深度（整数张量）
      - next_word: (B, T, depth_embd_dim) 表示当前新预测的 token（经过 MLP 得到的向量）
    输出：
      - 返回形状为 (B, T, T+1) 的 attachment logits（经过因果 mask 后）
    """

    def __init__(self, d_model, depth_embd_dim, max_depth=150, dropout=0.1):
        super(AttachmentHead, self).__init__()
        self.d_model = d_model # also for word emb
        self.depth_embd_dim = depth_embd_dim # dim of the embedding of words
        # 将输入 x 映射到 query 和 key（拼接后维度为 4*embd_dim）
        self.data_to_qk = nn.Linear(d_model, 4 * d_model) # FIXME: check the dim of q,k; in the original codebase it's 2*embd_dim
        # 用于计算 next_word 的 query 与 key 的 MLP，
        # 输入为拼接后的 [q, next_word]，输出维度为 embd_dim
        self.q_next_word_mlp = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(2 * d_model, 2 * d_model),
        )
        self.k_next_word_mlp = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(2 * d_model, 2 * d_model),
        )
        # 将 key 与堆栈深度信息拼接后映射回 embd_dim
        if dropout == 0:
            self.key_and_stack_mlp = nn.Sequential( # intermediate 0 (input): k & depth_embed == (B, T, T', 2*d_model + depth_embd_dim)
                nn.Linear(2 * d_model + depth_embd_dim, 2 * d_model + depth_embd_dim), # intermediate 1: still k & depth_embed: (B, T, T', 2*d_model + depth_embd_dim)
                nn.ReLU(), # intermediate 2: k & depth_embed: (B, T, T', 2*d_model + depth_embd_dim)
                # nn.Dropout(p=dropout), # NOTE: we don't need dropout here for saving memory
                nn.Linear(2 * d_model + depth_embd_dim, 2 * d_model), # intermediate 3 (out): k & depth_embed == (B, T, T', 2*d_model)
            )
        else:
            self.key_and_stack_mlp = nn.Sequential(
                nn.Linear(2 * d_model + depth_embd_dim, 2 * d_model + depth_embd_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(2 * d_model + depth_embd_dim, 2 * d_model),
            )
        # 堆栈深度嵌入
        self.beta = nn.Embedding(max_depth, depth_embd_dim)
        # 这里 bias 我们在 forward 中动态生成因果 mask
        
        # size of q/k/v is 2*d_model=D
        self.scale = 1 / ((2 * d_model) ** 0.5)
        self.max_stack_depth = max_depth


    def forward(self, x, stack_tape, next_word):
        # next_word 是当前预测的 token，形状为 (B, T, embd_dim = d_model now)
        # x 是隐藏状态序列，形状为 (B, T, d_model)

        B, T, C = x.size()
        # 得到 q 和 k，形状均为 (B, T, embd_dim)
        qk = self.data_to_qk(x)
        q, k = torch.split(qk, 2*self.d_model, dim=2) # (B, T, 2*d_model) each

        # 计算 next_word 的 query 与 key
        # 拼接 [q, next_word]，假设 next_word 的最后一维与 embd_dim 相同
        # print(q.shape)
        # print(next_word.shape)
        cat_inp = torch.cat([q, next_word], dim=-1)  # (B, T, 2*d_model)
        q = q.detach().cpu()
        next_word = next_word.detach().cpu()
        next_word_q = self.q_next_word_mlp(cat_inp)    # (B, T, 2*d_model)
        next_word_k = self.k_next_word_mlp(cat_inp)      # (B, T, 2*d_model)
        
        # D = 2*d_model

        # 将 k 扩展到每个目标位置： (B, T, D) -> (B, 1, T, D) -> (B, T, T, D)
        k = k.unsqueeze(1).expand(-1, T, -1, -1) # expanded_k
        # detach k
        # k = k.detach().cpu()
        

        # from utils import tensor_memory
        
        # print("Memory Allocated: ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
        # print("Memory Reserved: ", torch.cuda.memory_reserved() / 1024 / 1024, "MB")
        
        torch.cuda.empty_cache()
        if False:
            # TODO: gather to save gpu memory, 
            # TODO: but need to separate key_and_stack_mlp
            stack_tape = stack_tape.clamp(0, self.max_stack_depth - 1)
            d_emb_table = self.beta(torch.arange(self.max_stack_depth, device=stack_tape.device).int()) # shape [max_stack_depth, embd_dim]
            tabled_stack_info = d_emb_table.view(self.max_stack_depth, self.depth_embd_dim) # shape [max_stack_depth, embd_dim]
        else:
            # Compute the depth emb. 
            # stack_tape (B, T, T') -> (B, T, T', embd_dim=D)
            # clip stack_tape to [0, max_stack_depth-1]
            stack_tape = stack_tape.clamp(0, self.max_stack_depth - 1)
            depth_emb = self.beta(stack_tape.int())
            stack_tape = stack_tape.detach().cpu()
            # concatenate k and depth_emb
            # print("After depth emb net")
            # print("Memory Allocated: ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
            # print("Memory Reserved: ", torch.cuda.memory_reserved() / 1024 / 1024, "MB")
            # print("depth_emb:", tensor_memory(depth_emb))
            # print("k:", tensor_memory(k))
            k = self.key_and_stack_mlp(
                torch.cat([k, depth_emb], dim=-1)
            ) # (B, T, T', D+embd_dim) -> (B, T, T', D) 
            # print("k:", tensor_memory(k))
            torch.cuda.empty_cache()
            depth_emb = depth_emb.detach().cpu()
            # print("!! After key stack mlp")
            # print("Memory Allocated: ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
            # print("Memory Reserved: ", torch.cuda.memory_reserved() / 1024 / 1024, "MB")
            # 计算 attachment logits
            # next_word_q: (B, T, D)
            # k_with_info: (B, T, T', D)
            
            # attach_logits = (next_word_q.unsqueeze(
            #     2) @ k.transpose(-2, -1)).squeeze(2)  # (B, T, T')
            
            # use einsum?
            attach_logits = torch.einsum('bid,bijd->bij', next_word_q, k)  # (B, T, T')
                
            k = k.detach().cpu()
            attach_logits.mul_(self.scale) # (B, T, T)

        torch.cuda.empty_cache()
        # print("After attach logits computation")
        # print("k:", tensor_memory(k))
        # print("next_q:", tensor_memory(next_word_q))
        # print("attach_logits:", tensor_memory(attach_logits))
        # print("Memory Allocated: ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
        # print("Memory Reserved: ", torch.cuda.memory_reserved() / 1024 / 1024, "MB")
        # 计算 self-attachment 得分（no-reduce 情况）
        # next_word_k.mul_(self.scale) # (B, T, D)
        logits_self = (next_word_q.unsqueeze(
            2) @ next_word_k.unsqueeze(3)).squeeze(2)  # (B, T, 1)
        logits_self.mul_(self.scale) # (B, T, 1)

        # 在 attach_logits 最后拼接一列 0 值，使其形状变为 (B, T, T+1)
        
        # pad_tensor = torch.zeros(B, T, 1, device=attach_logits.device)
        # attach_logits_l = torch.cat(
        #     [attach_logits, pad_tensor], dim=-1)  # (B, T, T+1)
        attach_logits_l = torch.cat(
            [attach_logits, torch.zeros(B, T, 1, device=attach_logits.device)], dim=-1
        )  # (B, T, T+1)
        
        
        
        ### now insert logits_ self into the k +1 th position of attach_logits for each k
        # i.e. [n_batch, n_dest, n_src] -> [n_batch, n_dest, n_src + 1]
        # (B, T, 1) -> (B, T, T+1)
        
        # indices = (1 + torch.arange(T, device=attach_logits.device)
        #            ).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        # # 将 logits_self 插入到第 k+1 个位置
        
        # logits = attach_logits_l.scatter(
        #     2, indices, logits_self)  # (B, T, T+1)
        
        # indices = indices.detach().cpu()

        logits = attach_logits_l.scatter(
            2,
            (1 + torch.arange(T, device=attach_logits.device)
             .unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)),
            logits_self,
            # (B, T, 1) -> (B, T, T+1)
        )
        # import pdb
        # pdb.set_trace()


        # 在 logits 的第一行前填充一行全 0，使其形状变为 (B, T+1, T+1)
        # zeros_row = torch.zeros(B, 1, logits.size(2), device=logits.device)
        # logits = torch.cat([zeros_row, logits], dim=1)  # (B, T+1, T+1)
        logits = torch.cat(
            [torch.zeros(B, 1, logits.size(2), device=logits.device), logits], dim=1
        )  # (B, T+1, T+1)

        # 构造因果 mask：下三角 mask，形状为 (1, T+1, T+1)
        mask = torch.tril(torch.ones(
            T+1, T+1, device=logits.device)).unsqueeze(0)
        # be like
        # 1 0 0 0 0
        # 1 1 0 0 0
        # 1 1 1 0 0
        # 1 1 1 1 0
        # 1 1 1 1 1
        logits = logits.masked_fill(mask == 0, float("-inf"))
        mask = mask.detach().cpu()
        # import pdb;pdb.set_trace()

        # 去掉最上面一行，返回 (B, T, T+1)
        return logits[:, 1:]
        # 这里 T+1 是因为，原序列 T=x1...xT, xT 输入进去之后，如果选择 shift 则指向预测的 hat x T+1
        # logits[b, t, t+1] 表示在 t 时刻预测 hat x t+1, 选择 hat x t+1 作为 reduce (i.e. pure shift) 的概率


class PushdownTransformerConstituency(nn.Module):
    
    def __init__(self, vocab_size = 10000,
                 w_dim = 380,
                 n_head = 10,
                 d_head = 38,
                 d_inner = 900, # for FF
                 num_layers = 16,
                 dropout = 0.1,
                 dropoutatt = 0.0,
                 pad_id = 0,
                 bos_id = 1,
                 eos_id = 2,
                 stack_pad_id = -100,
                 pre_lnorm = False,
                 max_stack_depth = 150
                 ):
        super(PushdownTransformerConstituency, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = w_dim
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)

        self.emb = nn.Embedding(vocab_size, w_dim)
        self.emb_scale = w_dim ** 0.5
        self.projection = nn.Linear(w_dim, vocab_size)
        self.projection.weight = self.emb.weight

        self.num_layers = num_layers
        # self.w_dim = w_dim
            
        self.layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.layers.append(RelPartialLearnableDecoderLayer(n_head, w_dim, d_head, 
                                d_inner, dropout, dropoutatt, pre_lnorm=pre_lnorm))
            
        # PUSHDOWN LAYER FOR CONSTITUENCY
        self.max_depth = max_stack_depth
        self.pushdown_final_layer = RelPartialLearnablePushdownLayer(n_head, w_dim, d_head, d_inner, dropout, dropoutatt,
                                pre_lnorm=pre_lnorm, max_stack_depth=max_stack_depth)
        
        # ATTACHMENT HEAD
        self.attachment_head = AttachmentHead(d_model=w_dim, depth_embd_dim=d_head, max_depth=max_stack_depth, dropout=dropout)
        
        # POSITIONAL EMBEDDINGS FOR TXL STRUCTURES
        self.pos_emb = PositionalEmbedding(w_dim)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.stack_pad_id = stack_pad_id
        if self.num_layers > 5:
            logging.warning("You have set num_layers >> 1, please init the weights of the model (by kaiming fan in). If you have applied weight init, please ignore this warning.")
        
    # def init_weights(self):
    #     nn.init.normal_(self.r_w_bias, 0.0, 0.02)
    #     nn.init.normal_(self.r_r_bias, 0.0, 0.02)
    #     self.emb.weight.data.uniform_(-0.02, 0.02)
    #     self.projection.weight = self.emb.weight
    #     self.projection.bias.data.zero_()
    #     self.attachment_head.init_weights()
    #     for layer in self.layers:
    #         layer.init_weights()
    #     self.pushdown_final_layer.init_weights()

    def forward(self, 
                data, 
                target, 
                stack_tape,
                attachment_labels,
                mems=None, 
                return_h=False):
        
        # data: x, shape [B, T], ids of tokens seq[:T] where len(seq) = T+1
        # target: y = seq[1:T+1], shape [B, T], ids of tokens seq[1:T+1]
        # T+1: real length of seq, 
        # B: batch size
        # mems: None for now
        # stack_tape: [B, T(qlen), T(klen)] stack depth information
        # stack_tape[b, t, k] = d means that at time t (where we want to predict t+1), the stack depth of the k-th token is d
        # attachment_labels: [B, T] labels for attachment head, at [:,t] each range from 0 to t+1 (closed)
        # attachment_mask: [B, T, T+1] mask for attachment head

        if mems is not None:
            raise NotImplementedError("We ignore TXL mems for now.")
        # try transpose WITHIN the forward to let DataParallel be able to work
        data = data.transpose(0, 1) # [B, T] -> [T, B]
        target = target.transpose(0, 1) # [B, T] -> [T, B]
        # print(data, target)
        qlen, bsz = data.size()
        mlen = 0 if mems is None else mems[0].size(0)
        klen = qlen + mlen
        
        # WORD EMBEDDINGS
        word_emb = self.emb(data)
        
        # POSITIONAL EMBEDDINGS
        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                dtype=word_emb.dtype)
        pos_emb = self.pos_emb(pos_seq)
        
        # DECODER ATTENTION MASK
        
        dec_attn_mask = torch.triu(
            word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()[:,:,None] # FIXED: NOT .byte() ANYMORE but .bool()
        # import pdb;pdb.set_trace()
        # INITIAL HIDDEN STATE
        core_out = self.dropout(word_emb)
        pos_emb = self.dropout(pos_emb)
        
        # FORWARD PASS
        for layer in self.layers:
            core_out = layer.forward(core_out, pos_emb, self.r_w_bias,
                    self.r_r_bias, attn_mask=dec_attn_mask, mems=None)
            
        # PUSHDOWN LAYER PART
        core_out = self.pushdown_final_layer.forward(core_out, pos_emb, self.r_w_bias,
                    self.r_r_bias, stack_tape, attn_mask=dec_attn_mask, mems=None)
        
        # FINAL DROPOUT
        core_out = self.dropout(core_out) # shape [T, B, d_model] # h1, h2, ..., hT
        # PROJECTION
        logits = self.projection(core_out) # shape [T, B, vocab_size]    
        # ATTACHMENT HEAD
        # NOTE: Though in inference, we needs the logits to predict next_word, here we use target directly in training
        next_word = self.emb(target) # shape [T, B, embd_dim] # x2, x3, ..., xT+1

        # forward needs x shape (B, T, d_model), stack_tape shape (B, T, T), next_word shape (B, T, embd_dim)
        # print("ATTACHMENT HEAD")
        # print("Before")
        # print("Memory Allocated: ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
        # print("Memory Reserved: ", torch.cuda.memory_reserved() / 1024 / 1024, "MB")
        # print("In")
        attach_logits = self.attachment_head.forward(x = core_out.permute(1, 0, 2), stack_tape = stack_tape, next_word = next_word.permute(1, 0, 2)) # shape [B, T, T+1]
        # print("After")
        # print("Memory Allocated: ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
        # print("Memory Reserved: ", torch.cuda.memory_reserved() / 1024 / 1024, "MB")
        # r2, r3, ..., rT+1
        # attach_logits[b, t, t+1] 表示在 t 时刻预测下一个token: hat x t+1, 选择 hat x t+1 作为 reduce (i.e. pure shift) 的概率
        # attach_logits[b, t, k<=t] 表示在 t 时刻预测下一个token: hat x k, 选择 hat x k 作为 reduce (i.e. reduce + shift) 的概率
        # attach_logits shape [B, T, T+1]
        # for every sample, every t, logits over T+1 positions
        
        # mask the attach_logits with attachment_mask
        # print("ATTACH MASK", attachment_mask.shape)
        
        # logging.debug("ATTACH LABEL")
        # logging.debug(attachment_labels.shape)
        # logging.debug("ATTACH LOGITS")
        # logging.debug(attach_logits.shape)
        # logging.debug("SLICED 2D ATTACH LOGITS")
        # logging.debug(attach_logits[0, :, :].detach().cpu().numpy())
        # attach_logits = attach_logits.masked_fill(attachment_mask == 0, float("-inf")) # because we masked this in the forward pass of attachment_head
        # attach_logits = attach_logits.contiguous() # otherwise may cause error in view
        # attachment_labels = attachment_labels.contiguous()
        # attachment_labels: [B, T] labels for attachment head, so qlen+1 here is the number of classes
        loss_attach = F.cross_entropy(attach_logits.reshape(-1, qlen+1), attachment_labels.reshape(-1), ignore_index=self.stack_pad_id, reduction='sum')
        
        # needs: hidden (core_out) 1...k...T; stack tape; next_word (target) which is hat y, 2...T+1 (HERE WE START FROM 1) to input
        # needs stack labels (idxs to reduce with) for each time step and do the reduce
        # NOTE: if attach_logits[b, t, t+1] > 0, then shift, otherwise reduce
        # and make the logits cross-entropy the target
        
        # logging.debug("SLICED 2D LOGITS")
        # logging.debug(logits[0, :, :].detach().cpu().numpy())
        # logits = logits.contiguous()
        # target = target.contiguous()
        loss_words = F.cross_entropy(logits.reshape(-1, self.vocab_size), target.reshape(-1), ignore_index=self.pad_id, reduction='sum')
        # loss = loss_words + loss_attach
        
        if return_h:
            return loss_words, loss_attach, core_out
        else:
            return loss_words, loss_attach
        
    def take_step_silver_tree(self, 
                            ids, # shape [qlen+1, bsz]. the real seq is [qlen + 1, bsz] while first is bos and last is eos
                            stack_tape, # shape [bsz, step, step]
                            list_reduced, # list of sets of reduced tokens, len = bsz, each set: reduced tokens before -> set as -inf in attach logits
                            step
                            ):
    
        # NOTE: take step w/ stack tape, and predict a step of attachment while the word seq is given gold
        # at least the src is [:1] = [0] = bos
        # final: next_tgt is eos, i.e. [step] is [T]
        
        # in step, want to predict the states in step+1
        # ids: [bsz, qlen+1]
        # src = ids.transpose(0, 1)[:step+1]
        # tgt = ids.transpose(0, 1)[step+1]
        src, next_tgt, tgt = ids.transpose(0, 1)[:step], ids.transpose(0, 1)[step], ids.transpose(0, 1)[1:step+1]
        
        word_emb = self.emb(src) # shape [T, B, d_model]
        pos_seq = torch.arange(src.size(0)-1, -1, -1.0, device=word_emb.device, 
                                dtype=word_emb.dtype)
        pos_emb = self.pos_emb(pos_seq)
        dec_attn_mask = torch.triu(
            word_emb.new_ones(src.size(0), src.size(0)), diagonal=1).bool()[:,:,None]
        core_out = self.dropout(word_emb)
        pos_emb = self.dropout(pos_emb)
        for layer in self.layers:
            core_out = layer.forward(core_out, pos_emb, self.r_w_bias,
                    self.r_r_bias, attn_mask=dec_attn_mask, mems=None)
        core_out = self.pushdown_final_layer.forward(core_out, pos_emb, self.r_w_bias,
                    self.r_r_bias, stack_tape, attn_mask=dec_attn_mask, mems=None)
        core_out = self.dropout(core_out)
        logits_next_word = self.projection(core_out[-1]) # shape [B, vocab_size]
        # log softmax
        logits_next_word = F.log_softmax(logits_next_word, dim=-1)
        # the logprobs of the next word (in next_tgt)
        # use gather
        # next_tgt shape: [B]
        logprobs_next_word_tgt = torch.gather(logits_next_word, 1, next_tgt.unsqueeze(1)).squeeze(1) # shape [B]
        
        # for attachment
        next_word = self.emb(tgt)
        attach_logits = self.attachment_head.forward(x = core_out.permute(1, 0, 2), stack_tape = stack_tape, next_word = next_word.permute(1, 0, 2))
        logits_next_attach = attach_logits[:, -1, :].squeeze(1) # shape [B, T+1]
        logits_next_attach = F.log_softmax(logits_next_attach, dim=-1)
        # postprocess logits_next_attach
        for batch_idx, reduced_set in enumerate(list_reduced):
            for reduced_pos in reduced_set:
                logits_next_attach[batch_idx, reduced_pos] = -float('inf')


        # logprobs_next_word_tgt: the log prob of the next word (exactly that word, not a prob distribution), in a batch
        # logits_next_attach: the log probS of the next attachment (which IS a prob distribution so we want to SELECT on this), in a batch
        return logprobs_next_word_tgt, logits_next_attach
    