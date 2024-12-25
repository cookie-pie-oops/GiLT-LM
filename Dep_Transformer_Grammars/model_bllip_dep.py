import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from masking_bllip import utils as masking_utils
from masking_bllip import masking_types as types
import time
from helping_utils.logger import configure_logger, get_logger
logger = get_logger()

class MLPforBiaffine(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPforBiaffine, self).__init__()

        self.L_1 = nn.Linear(input_dim, input_dim)
        self.L_2 = nn.Linear(input_dim, output_dim)

        nn.init.kaiming_uniform_(self.L_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.L_1.bias)
        nn.init.kaiming_uniform_(self.L_2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.L_2.bias)

        self.elu = nn.ELU()
    
    def forward(self, x):
        x = self.L_1(x)
        x = self.elu(x)
        x = self.L_2(x)
        x = self.elu(x)
        return x

class BiaffineAttention(nn.Module):
    def __init__(self, d_model, input_dim, type="default"):
        super(BiaffineAttention, self).__init__()

        self.input_dim = input_dim

        self.W = nn.Linear(input_dim, input_dim, bias=False)
        self.V = nn.Parameter(torch.Tensor(1, input_dim))
        self.U = nn.Parameter(torch.Tensor(1, input_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.type = type

        # self.f_1 = MLPforBiaffine(d_model, input_dim)
        # self.f_2 = MLPforBiaffine(d_model, input_dim)
        if type == "default":
            self.softmax = nn.Softmax(dim=-1)
        elif type == "Multi":
            self.root_representation = nn.Parameter(torch.Tensor(1, input_dim))
            self.softmax = nn.Sigmoid()

        # nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.root_representation)
        nn.init.zeros_(self.bias)
    
    def forward(self, input1, input2): # 1*d, l*d
        # input1 = self.f_1(input1) #1*d
        # input2 = self.f_2(input2) #l*d
        if self.type == "Multi":
            B = input2.size(0)
            root_tokens = self.root_representation.repeat(B, 1, 1)
            input2 = torch.cat((root_tokens, input2), dim=1)

        H_2 = self.W(input2) # d*l
        score = torch.matmul(input1, H_2.transpose(1,2))  # b1d * bdl = b1l

        score += torch.matmul(self.U, input1.transpose(1,2))  # 1d * bd1 = b11
        score += torch.matmul(self.V, input2.transpose(1,2))  # 1d * bdl = b1l

        score += self.bias  # b,1,l
        score = score.to(torch.float64)
        score = self.softmax(score)   
        return score

class PositionalEmbedding(nn.Module):
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
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)
        # self.layer_norm = nn.Identity()
        self.pre_lnorm = pre_lnorm

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
    def __init__(self, n_head, d_model, d_head, dropout, tgt_len = None, 
                                ext_len = None, mem_len = None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Sequential(
            nn.Linear(d_model, 3 * n_head * d_head, bias=False),
            nn.Dropout(dropout)
            )

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropout)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)
        # self.layer_norm = nn.Identity()
        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

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

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, attn_relpos=None, min_len=None, max_len=None
        , mems=None, terminal=False, past_keys=None, past_values=None, cache=False, use_graph=False):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)  # L, M-m, B
        # print(qlen, rlen)
        # r: M-m * None * d_model
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r) # M-m * None * (n_head * d_head) // M-m * B * (n_head * d_head)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                # print(w.shape)
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)
            
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            # _, w_head_k, _ = torch.chunk(w_graphlayer, 3, dim=-1)
        # #test    
        # r_heads = self.qkv_net(r)
        # r_head_q, r_head_k, r_head_v = torch.chunk(r_heads, 3, dim=-1)
        # r_head_q = r_head_q.view(rlen, self.n_head, self.d_head)
        # r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)
        # #---
        if cache:
            new_key = w_head_k.view(qlen, bsz, -1)
            new_value = w_head_v.view(qlen, bsz, -1)
        
        if past_keys is not None:
            w_head_k = torch.cat([past_keys, w_head_k], dim=0)
            w_head_v = torch.cat([past_values, w_head_v], dim=0)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head

        # if composed and rlen == qlen:
        #     r_head_k = r_head_k.view(rlen, bsz, self.n_head, self.d_head)       # rlen x bsz x n_head x d_head
        # else:
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # rlen x n_head x d_head
        # #test
        # r_w_bias = r_head_q[-1]
        # r_r_bias = r_head_q[-1]
        # #---
        #### compute attention score
        rw_head_q = w_head_q + r_w_bias # L * B * n_head * d_head               # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        # if composed and rlen == qlen:
        #     BD = torch.einsum('ibnd,jbnd->ijbn', (rr_head_q, r_head_k))         # qlen x rlen x bsz x n_head
        # else:
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x rlen x bsz x n_head
        # logger.info("BD: %s", str(BD.shape))
        if attn_relpos is None or use_graph:
            if use_graph:
                attn_relpos = torch.clip(attn_relpos, min_len, max_len).long()
                # attn_relpos = torch.zeros_like(attn_relpos)
                attn_relpos = (max_len - attn_relpos).long()
                attn_relpos = attn_relpos.permute(1, 2, 0)
                r_squeezed = r.squeeze(1)
                d = r_squeezed[attn_relpos]
                w2 = w.unsqueeze(1)
                w2 = w2.repeat(1, qlen, 1, 1)
                w_graphlayer = self.qkv_net(w2 + d)

                _, w_k_graphlayer, _ = torch.chunk(w_graphlayer, 3, dim=-1)
                w_k_graphlayer = w_k_graphlayer.view(klen, rlen, bsz, self.n_head, self.d_head)
                # r_head_k = r_head_k.unsqueeze(1).repeat(1,bsz,1,1)
                # r_head_k = r_head_k + w_k_graphlayer
                Moderate_BD = torch.einsum('ibnd,ijbnd->ijbn',(w_head_q, w_k_graphlayer))
                BD = Moderate_BD + BD
            BD = self._rel_shift(BD)
        else:
            # BD = self._rel_shift(BD)
            attn_relpos = torch.clip(attn_relpos, min_len, max_len).long()
            # attn_relpos = torch.randint(min_len, max_len, size=attn_relpos.shape).long().cuda()
            # attn_relpos = 10 * torch.ones_like(attn_relpos)
            attn_relpos = (max_len - attn_relpos).long()
            attn_relpos = attn_relpos.permute(1, 2, 0)
            BD = self._rel_shift(BD) + BD.gather(1, attn_relpos.unsqueeze(-1).expand(-1, -1, -1, BD.shape[-1]))
            # BD = torch.einsum('ijbn,bisj->isbn', BD, relpos_one_hot)  
                          # qlen x klen x bsz x n_head
        # logger.info("AC: %s", str(AC.shape))
        # logger.info("BD: %s", str(BD.shape))
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        
        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    ~attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    ~attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)
                
        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.drop(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
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
        
        if  cache:
            return output, new_key, new_value
        else:
            return output

class TransformerGrammarLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropoutf, dropouta,
                 **kwargs):
        super(TransformerGrammarLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropouta, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropoutf, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))
    def forward(self, dec_inp, r, r_w_bias, r_r_bias, attn_mask=None, attn_relpos=None, min_len=None, max_len=None, mems=None, terminal=False, past_keys=None, past_values=None, cache=False, use_graph=False):
        if cache:
            output, new_key, new_value = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                                attn_mask=attn_mask, attn_relpos=attn_relpos,
                                min_len=min_len, max_len=max_len, mems=mems, terminal=terminal, past_keys=past_keys, past_values=past_values, cache=cache, use_graph=use_graph)
            output = self.pos_ff(output)

            return output, new_key, new_value
        else:
            output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                                attn_mask=attn_mask, attn_relpos=attn_relpos,
                                min_len=min_len, max_len=max_len, mems=mems, terminal=terminal, past_keys=past_keys, past_values=past_values, use_graph=use_graph)
            output = self.pos_ff(output)
            
            return output

class TransformerGrammar(nn.Module):
    def __init__(self, vocab_size = 10000,
                 w_dim = 380,
                 n_head = 10,
                 d_head = 38,
                 d_inner = 900,
                 num_layers = 16,
                 dropout = 0.1,
                 dropoutatt = 0.0,
                 pad_id = 0,
                 bos_id = 1,
                 eos_id = 2,
                 left_arc = None,
                 right_arc = None,
                 left_arc2 = None,
                 right_arc2 = None,
                 startofword_id = [],
                 pre_lnorm = False):
        super(TransformerGrammar, self).__init__()
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
        self.w_dim = w_dim
            
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(TransformerGrammarLayer(n_head, w_dim, d_head, 
                                d_inner, dropout, dropoutatt, tgt_len = None, 
                                ext_len = None, mem_len = None,
                                pre_lnorm = pre_lnorm))
        
        self.pos_emb = PositionalEmbedding(w_dim)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.left_arc = left_arc
        self.right_arc = right_arc
        self.startofword_id = startofword_id
        self.left_arc2 = left_arc2
        self.right_arc2 = right_arc2

    def forward(self, x, startofword_x, length, use_mask=None, document_level=False, return_h=False,
        max_relative_length=None, min_relative_length=None, iseval=False, sents_index_to_id=None, sents_arrow=None):
        
        attn_mask = []
        attn_relpos = []
        inputs = []
        targets = []
        batch = len(x)
        if use_mask is None or use_mask == "None":
            length_i = max([len(sent) for sent in x])
            for sent in x:
                src_ = sent[:-1]
                tgt_ = sent[1:]
                src_p = src_ + [self.pad_id] * (length_i - len(src_))
                inputs.append(np.array(src_p))
                tgt_p = tgt_ + [self.pad_id] * (length_i - len(tgt_))
                targets.append(np.array(tgt_p))
            if torch.cuda.is_available():
                inputs = torch.LongTensor(np.array(inputs)).cuda()
                targets = torch.LongTensor(np.array(targets)).cuda()
                attn_mask = torch.tril(torch.ones((length_i, length_i), dtype = torch.uint8)).cuda().bool()
            else:
                inputs = torch.LongTensor(np.array(inputs))
                targets = torch.LongTensor(np.array(targets))
                attn_mask = torch.tril(torch.ones((length_i, length_i), dtype = torch.uint8)).bool() 

            attn_mask = attn_mask.unsqueeze(0).expand(batch, -1, -1)
            attn_relpos = None

        elif use_mask == 'graphlayer':
            max_graphlayer_rel = 0
            length_i = max([len(sent) for sent in x])
            for sent in x:
                src_ = sent[:-1]
                tgt_ = sent[1:]
                src_p = src_ + [self.pad_id] * (length_i - len(src_))
                inputs.append(np.array(src_p))
                tgt_p = tgt_ + [self.pad_id] * (length_i - len(tgt_))
                targets.append(np.array(tgt_p))
            if torch.cuda.is_available():
                inputs = torch.LongTensor(np.array(inputs)).cuda()
                targets = torch.LongTensor(np.array(targets)).cuda()
                attn_mask = torch.tril(torch.ones((length_i, length_i), dtype = torch.uint8)).cuda().bool()
            else:
                inputs = torch.LongTensor(np.array(inputs))
                targets = torch.LongTensor(np.array(targets))
                attn_mask = torch.tril(torch.ones((length_i, length_i), dtype = torch.uint8)).bool() 

            attn_mask = attn_mask.unsqueeze(0).expand(batch, -1, -1)

            left_sents_arrow, right_sents_arrow = sents_arrow

            attn_relpos = []
            for left_sent_arrow, right_sent_arrow in zip(left_sents_arrow, right_sents_arrow):
                Degree_out = [0]*(length_i + 1)
                Degree_in = [0]*(length_i + 1)
                graph = np.zeros((length_i + 1, length_i + 1))
                sent_attn_relpos = []
                for i in range(len(left_sent_arrow)):
                    if left_sent_arrow[i]:  #i <- j
                        for j in left_sent_arrow[i]:
                            graph[j][i + 1] = 1
                            Degree_out[j - 1] += 1
                            Degree_in[i] += 1
                    if right_sent_arrow[i]: #i -> j
                        for j in right_sent_arrow[i]:
                            graph[i + 1][j] = 1
                            Degree_in[j - 1] += 1
                            Degree_out[i] += 1
                    rel_pos_1 = Degree_out[:-1]
                    rel_pos_2 = Degree_in[:-1]
                    # Wk R
                    rel_pos_1 = rel_pos_1[(i+1):] + rel_pos_1[:(i+1)]
                    rel_pos_2 = rel_pos_2[(i+1):] + rel_pos_2[:(i+1)]
                    sent_attn_relpos.append(copy.deepcopy(10 * np.array(rel_pos_1) + 1 * np.array(rel_pos_2)))
                max_graphlayer_rel = max(Degree_out[:-1])
                for j in range(length_i - len(sent_attn_relpos)):
                    sent_attn_relpos.append([0]*(length_i))
                attn_relpos.append(sent_attn_relpos)
                # attn_relpos.append(10 * np.array(Degree_out[:-1]) + 1 * np.array(Degree_in[:-1]))
            
            attn_relpos = torch.LongTensor(np.array(attn_relpos))
            if torch.cuda.is_available():
                attn_relpos = attn_relpos.cuda()

        elif use_mask == 'sdp_arc':
            assert sents_index_to_id is not None and sents_arrow is not None
            index_dict = {"bos_id": self.bos_id, "vocab_size": self.vocab_size, "pad_id": self.pad_id,
                "left_arc": self.left_arc, "right_arc": self.right_arc, "left_arc2": self.left_arc2, "right_arc2": self.right_arc2}
            length_i = max([len(sent) for sent in x])
            for sent, sent_startofword, sent_index_to_id, sent_arrow in zip(x, startofword_x, sents_index_to_id, sents_arrow):
                id_to_index = {}
                for i in range(len(sent_index_to_id)):
                    if sent_index_to_id[i] != -1 and sent_index_to_id[i] not in id_to_index:
                        id_to_index[sent_index_to_id[i]] = i
                
                # src_ = torch.LongTensor(sent[:-1])
                # tgt_ = torch.LongTensor(sent[1:])
                # src_startofword = torch.LongTensor(sent_startofword[:-1])
                # tgt_startofword = torch.LongTensor(sent_startofword[1:])
                src_ = sent[:-1]
                tgt_ = sent[1:]
                src_p = src_ + [self.pad_id] * (length_i - len(src_))
                tgt_p = tgt_ + [self.pad_id] * (length_i - len(tgt_))

                mask = torch.tril(torch.ones((len(src_p), len(src_p)), dtype = torch.uint8)).bool()
                Tree_structure = masking_utils.UnionFind(len(src_p))
                false_position_list = []
                for i in range(len(src_p)):
                    if src_p[i] == self.pad_id:
                        break
                    if src_p[i] == self.left_arc or src_p[i] == self.right_arc:
                        mask[i][:] = False
                        arrow_word_start = id_to_index[sent_arrow[i]]
                        arrow_words_end = arrow_word_start + 1
                        while sent_index_to_id[arrow_words_end] == sent_arrow[i]:
                            arrow_words_end += 1
                        mask[i][arrow_word_start:arrow_words_end] = True

                        pos = i - 1
                        while sent_index_to_id[pos] == -1 and pos not in id_to_index.values():
                            pos -= 1
                        end_predicate = pos + 1
                        while pos not in id_to_index.values():
                            pos -= 1
                        start_predicate = pos

                        Tree_structure.attn_bool[arrow_word_start:arrow_words_end] = [False] * (arrow_words_end - arrow_word_start)
                        Tree_structure.attn_bool[start_predicate:end_predicate] = [False] * (end_predicate - start_predicate)
                        # Tree_structure.union(i, id_to_index[sent_arrow[i]]) #arc token and parent token also set False
                        # for idx in range(start_predicate, end_predicate, 1):
                            # Tree_structure.union(idx, id_to_index[sent_arrow[i]]) #compose here and use it later 
                        mask[i][start_predicate:end_predicate] = True
                        if src_p[i] == self.left_arc:
                            predicate_id = [k for k, v in id_to_index.items() if v == pos][0]
                            id_to_index[predicate_id] = i
                        else:
                            id_to_index[sent_arrow[i]] = i
                        mask[i][i] = True
                    elif src_p[i] == self.left_arc2 or src_p[i] == self.right_arc2:
                        false_position_list = Tree_structure.Get_false_position(i)
                        Tree_structure.attn_bool[i] = False
                        for false_pos in false_position_list:
                            mask[i][false_pos] = False
                        false_position_list.append(i)
                    else:
                        for false_pos in false_position_list:
                            mask[i][false_pos] = False
                
                attn_mask.append(np.array(mask))
                inputs.append(np.array(src_p))
                targets.append(np.array(tgt_p))

            if torch.cuda.is_available():
                inputs = torch.LongTensor(np.array(inputs)).cuda()
                targets = torch.LongTensor(np.array(targets)).cuda()
                attn_mask = torch.LongTensor(np.array(attn_mask)).cuda().bool()
            else:
                inputs = torch.LongTensor(np.array(inputs))
                targets = torch.LongTensor(np.array(targets))
                attn_mask = torch.LongTensor(np.array(attn_mask)).bool()
            
            attn_relpos = None
        
        elif use_mask == 'txl' or use_mask == 'txl_arc':
            length_i = max([len(sent) for sent in x])
            ranges = masking_utils.TokenTypeRanges(self.bos_id, self.pad_id, self.vocab_size, self.left_arc, self.right_arc)
            maskrules = masking_utils.get_masking_rules(
                "stack_compose_double_closing_nt", #txl
                sequence_length=512, 
                memory_length=512, 
                transparency_prob=0.0,
                gather_into_new_memory=True, 
                transparency_depth_threshold=-1 
            )
            for sent, sent_startofword in zip(x, startofword_x):
                src_ = torch.LongTensor(sent[:-1])
                tgt_ = torch.LongTensor(sent[1:])
                src_startofword = torch.LongTensor(sent_startofword[:-1])
                # print(src_startofword)
                tgt_startofword = torch.LongTensor(sent_startofword[1:])
                info_tuple = masking_utils.compute_token_types(
                    {"inputs": src_, "labels": tgt_}, ranges
                )
                startofword_tuple = masking_utils.compute_token_types(
                    {"inputs": src_startofword, "labels": tgt_startofword}, ranges
                )
                # print(startofword_tuple['inputs_ttypes'])
                chunks = maskrules.chunks_for_sequence(info_tuple['inputs'], startofword_tuple['inputs_ttypes'],
                                                       info_tuple['labels'], startofword_tuple['labels_ttypes'])
                chunks = [types.Chunk(None, *chunk) for chunk in chunks]
                if not document_level:
                    chunk = chunks[0]
                    src_p = chunk.inputs[:length-1]
                    composed_pos = chunk.composed_position[:length-1]
                    src_raw = sent[:]
                    idx = 0
                    if use_mask != 'txl_arc':
                        for i in range(len(sent)):
                            if sent[i] == self.left_arc or sent[i] == self.right_arc:
                                src_raw[i] = src_p[composed_pos[idx]]
                                idx += 1
                            idx += 1
                    src_ = src_raw[:-1]
                    tgt_ = sent[1:]
                    src_p = src_ + [self.pad_id] * (length_i - len(src_))
                    inputs.append(np.array(src_p))
                    tgt_p = tgt_ + [self.pad_id] * (length_i - len(tgt_))
                    targets.append(np.array(tgt_p))
            inputs = torch.LongTensor(np.array(inputs))  #!!cuda
            targets = torch.LongTensor(np.array(targets))  #!!cuda

            attn_mask = torch.tril(torch.ones((length_i, length_i), dtype = torch.uint8)).bool()  #!!cuda
            attn_mask = attn_mask.unsqueeze(0).expand(batch, -1, -1)
            attn_relpos = None

        else:
            ranges = masking_utils.TokenTypeRanges(self.bos_id, self.pad_id, self.vocab_size, self.left_arc, self.right_arc)
            maskrules = masking_utils.get_masking_rules(
                "stack_compose_double_closing_nt", 
                sequence_length=512, 
                memory_length=512, 
                transparency_prob=0.0,
                gather_into_new_memory=True, 
                transparency_depth_threshold=-1 
            )
            for sent, sent_startofword in zip(x, startofword_x):
                src_ = torch.LongTensor(sent[:-1])
                # print(src_)
                tgt_ = torch.LongTensor(sent[1:])
                src_startofword = torch.LongTensor(sent_startofword[:-1])
                # print(src_startofword)
                tgt_startofword = torch.LongTensor(sent_startofword[1:])
                info_tuple = masking_utils.compute_token_types(
                    {"inputs": src_, "labels": tgt_}, ranges
                )
                startofword_tuple = masking_utils.compute_token_types(
                    {"inputs": src_startofword, "labels": tgt_startofword}, ranges
                )
                # print(startofword_tuple['inputs_ttypes'])
                chunks = maskrules.chunks_for_sequence(info_tuple['inputs'], startofword_tuple['inputs_ttypes'],
                                                       info_tuple['labels'], startofword_tuple['labels_ttypes'])
                chunks = [types.Chunk(None, *chunk) for chunk in chunks]
                if not document_level:
                    # only consider the first chunk
                    chunk = chunks[0]
                    src_p = chunk.inputs[:length-1]
                    # print(src_p)
                    composed_pos = chunk.composed_position[:length-1]
                    if use_mask != 'arc':
                        src_p = src_p[composed_pos]
                    # print(src_p)
                    inputs.append(np.array(src_p))
                    tgt_p = chunk.labels[:length-1]
                    # new_length, = np.where(tgt_p == self.pop_root)
                    # new_length = new_length[0]
                    # new_length += 1
                    targets.append(np.array(tgt_p))
                    mask = chunk.attn_mask[:length-1, :length-1]
                    # with np.printoptions(threshold=np.inf):
                    #     print(mask[:new_length, :new_length])
                    for i in range(len(mask)):
                        mask[i, i] = 1
                    attn_mask.append(np.array(mask))
                    chunk_len = len(chunk.attn_mask[0])
                    relpos = chunk.attn_relpos[:len(mask), chunk_len:chunk_len + len(mask)]
                    if use_mask == 'new':
                        relpos = np.clip(relpos, -1, 0)
                    # with np.printoptions(threshold=np.inf):
                    #     print(relpos[:new_length, :new_length])
                    # exit()
                    attn_relpos.append(np.array(relpos))
                else:
                    pass #remain to be implemented
            inputs = torch.LongTensor(np.array(inputs)).cuda()
            targets = torch.LongTensor(np.array(targets)).cuda()
            attn_mask = torch.LongTensor(np.array(attn_mask)).cuda().bool()
            attn_relpos = torch.LongTensor(np.array(attn_relpos)).cuda()

        if use_mask == 'linear':
            attn_relpos = None
        inputs = inputs.permute(1, 0).contiguous()
        targets = targets.permute(1, 0).contiguous()
        attn_mask = attn_mask.permute(1, 2, 0).contiguous()

        seq_len = inputs.size(0)

        word_emb = self.emb(inputs)
        
        if use_mask == None or use_mask == 'txl' or use_mask == 'txl_arc' or use_mask == 'linear' or use_mask == "None" or use_mask == 'sdp_arc' or use_mask == 'graphlayer':
            pos_emb = self.pos_emb(torch.arange(seq_len-1, -1, -1.0, device=word_emb.device))
            max_relative_length = seq_len - 1
            min_relative_length = 0
        else:
            if max_relative_length is None:
                max_relative_length = seq_len
            if min_relative_length is None:
                min_relative_length = -seq_len
            else:
                min_relative_length = min_relative_length - 1
            pos_emb = self.pos_emb(torch.arange(max_relative_length, min_relative_length, -1.0, device=word_emb.device))
        
        core_out = self.dropout(word_emb)
        pos_emb = self.dropout(pos_emb)
        hiddens = []
        hiddens.append(core_out)
        for i, layer in enumerate(self.layers):
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, attn_mask=attn_mask,
                attn_relpos=attn_relpos, min_len=min_relative_length, max_len=max_relative_length, use_graph=True)
            hiddens.append(core_out)
            if i < len(self.layers) - 1:
                core_out = self.dropout(core_out)
        core_out = self.dropout(core_out)

        logits = self.projection(core_out) 
        crit = nn.CrossEntropyLoss(reduction='none', ignore_index=self.pad_id)
        prob = logits.view(seq_len, batch, -1)
        prob = prob.permute(0, 2, 1)
        loss = crit(prob, targets)
        
        # if iseval:
        #     pred_seq = np.argmax(prob.cpu(), axis=1).permute(1,0)
        #     tgt_seq = targets.permute(1, 0).contiguous()
        #     import sentencepiece as spm
        #     sp = spm.SentencePieceProcessor()
        #     sp.load('../data_process/spm_parsing/BLLIP_spm.model')
        #     for i in range(len(pred_seq)):
        #         if 2 in pred_seq[i]:
        #             index = torch.where(pred_seq[i] == 2)[0][0]
        #         print("\npred seq:", sp.decode(pred_seq[i][:index].tolist()))
        #         print("gt seq:", sp.decode(tgt_seq[i].tolist()))
        #         import pdb; pdb.set_trace()

        loss = loss.permute(1, 0).contiguous()
        # logger.info(targets[:, -1])
        # logger.info(loss[-1])
        # logger.info(torch.sum(loss[-1]) / targets.size(0))
        # word_loss = loss[-1][(targets[:, -1] != self.pad_id) & (targets[:, -1] != self.left_arc) & (
        #     targets[:, -1] != self.right_arc) & (targets[:, -1] != self.left_arc2) & (
        #     targets[:, -1] != self.right_arc2) ] 
        # word_loss = loss[-1][(targets[:, -1] != self.pad_id) & (targets[:, -1] != self.left_arc) & (
        #     targets[:, -1] != self.right_arc)]
        # logger.info(loss[-1][])
        # logger.info(avg_loss)
        # logger.info(avg_loss_2)
        # exit()
        loss = loss.sum(1)
        
        # return word_loss

        if return_h:
            return loss, core_out
        else:
            return loss


    def constrained_forward_gen(self, 
                        new_token: torch.Tensor,
                        new_token_2: torch.Tensor,
                        token_mask: torch.Tensor,
                        past_keys: torch.Tensor, # batch * L-1 * (layer * dim)
                        past_values: torch.Tensor,
                        # padding_lengths: torch.Tensor,
                        attn_masks: torch.Tensor,
                        relative_pos: torch.Tensor,
                        max_relative_length: int,
                        min_relative_length: int,
                        # finished_beam_mask: torch.Tensor,
                        use_mask: str
                        ):
        with torch.no_grad():
            new_token = new_token.permute(1, 0).contiguous() # 1 * batch
            new_token_2 = new_token_2.permute(1, 0).contiguous()
            token_mask = token_mask.permute(1, 0).contiguous()
            attn_masks = attn_masks.permute(1, 2, 0).contiguous() # 1 * L * batch
            
            past_keys_p = None
            past_values_p = None
            if past_keys is not None:
                past_keys_p = past_keys.reshape(past_keys.size(0), past_keys.size(1), self.num_layers, self.w_dim)
                past_values_p = past_values.reshape(past_values.size(0), past_values.size(1), self.num_layers, self.w_dim)
                past_keys_p = past_keys_p.permute(2, 1, 0, 3).contiguous() # layer * L-1 * batch * w_dim
                past_values_p = past_values_p.permute(2, 1, 0, 3).contiguous() 

            word_emb = self.emb(new_token)
            word_emb += self.emb(new_token_2) * token_mask.unsqueeze(-1)
            
            batch = new_token.size(1)
            seq_len = attn_masks.size(1)

            if not use_mask.startswith('txl'):
                pos_emb = self.pos_emb(torch.arange(max_relative_length, min_relative_length - 1, -1.0, device=word_emb.device))
            else:
                pos_emb = self.pos_emb(torch.arange(seq_len - 1, -1, -1.0, device = word_emb.device))
                relative_pos = None
            hiddens = []
            core_out = word_emb
            # hiddens.append(core_out)

            new_keys = torch.full((self.num_layers, 1, batch, self.w_dim), 0.0, device=word_emb.device)
            new_values = torch.full((self.num_layers, 1, batch, self.w_dim), 0.0, device=word_emb.device)
            
            for i, layer in enumerate(self.layers):
                if past_values_p is not None:
                    core_out, new_key, new_value = \
                                    layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, 
                                    attn_mask=attn_masks, attn_relpos=relative_pos, 
                                    min_len=min_relative_length, max_len=max_relative_length, 
                                    past_keys=past_keys_p[i], past_values=past_values_p[i], cache=True)
                else:
                    core_out, new_key, new_value = \
                                    layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, 
                                    attn_mask=attn_masks, attn_relpos=relative_pos, 
                                    min_len=min_relative_length, max_len=max_relative_length, 
                                    past_keys=None, past_values=None, cache=True)
                # hiddens.append(core_out)
                new_keys[i] = new_key
                new_values[i] = new_value

            logits = self.projection(core_out) 
            prob = logits.view(1, batch, -1)
            prob = prob.log_softmax(-1)
            new_keys = new_keys.permute(2, 1, 0, 3).contiguous()
            new_values = new_values.permute(2, 1, 0, 3).contiguous()
            new_keys = new_keys.reshape(new_keys.size(0), new_keys.size(1), -1)
            new_values = new_values.reshape(new_values.size(0), new_values.size(1), -1)

            return prob, new_keys, new_values

                