import copy
import heapq
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from masking_bllip import utils as masking_utils
from masking_bllip import masking_types as types
import time
from helping_utils.logger import configure_logger, get_logger
logger = get_logger()


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def find_label_idx(sent, prefix):
    main_str = ",".join(map(str, sent))
    sub_str = ",".join(map(str, prefix))
    idx = main_str.find(sub_str)
    if idx == -1:
        return -1
    return main_str[:idx].count(",") if idx != 0 else 0

class FF_llama(nn.Module):
    def __init__(self, d_model, d_inner):
        super(FF_lamma, self).__init__()
        d_inner = int(2 * d_inner / 3)

        self.w1 = nn.Linear(d_model, d_inner)
        self.w2 = nn.Linear(d_inner, d_model)
        self.w3 = nn.Linear(d_model, d_inner)

        self.layer_norm = RMSNorm(d_model)

    def forward(self, inp):
        x = self.layer_norm(inp)
        return inp + self.w2(F.silu(self.w1(x)) * self.w3(x))


# class Attention_llama(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
#         self.n_local_heads = args.n_heads // model_parallel_size
#         self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
#         self.n_rep = self.n_local_heads // self.n_local_kv_heads
#         self.head_dim = args.dim // args.n_heads

#         self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
#         self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

#         self.cache_k = torch.zeros(
#             (
#                 args.max_batch_size,
#                 args.max_seq_len,
#                 self.n_local_kv_heads,
#                 self.head_dim,
#             )
#         ).cuda()
#         self.cache_v = torch.zeros(
#             (
#                 args.max_batch_size,
#                 args.max_seq_len,
#                 self.n_local_kv_heads,
#                 self.head_dim,
#             )
#         ).cuda()

#     def forward(
#         self,
#         x: torch.Tensor,
#         start_pos: int,
#         freqs_cis: torch.Tensor,
#         mask: Optional[torch.Tensor],
#     ):
#         bsz, seqlen, _ = x.shape

#         xq, xk, xv = torch.chunk(self.qkv_net(x), 3, dim=-1)
#         xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
#         xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
#         xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

#         xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

#         self.cache_k = self.cache_k.to(xq)
#         self.cache_v = self.cache_v.to(xq)

#         self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
#         self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

#         keys = self.cache_k[:bsz, : start_pos + seqlen]
#         values = self.cache_v[:bsz, : start_pos + seqlen]

#         # repeat k/v heads if n_kv_heads < n_heads
#         keys = repeat_kv(
#             keys, self.n_rep
#         )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
#         values = repeat_kv(
#             values, self.n_rep
#         )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

#         xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
#         keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
#         values = values.transpose(
#             1, 2
#         )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
#         scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
#         if mask is not None:
#             scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
#         scores = F.softmax(scores.float(), dim=-1).type_as(xq)
#         output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
#         output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
#         return self.wo(output)

def dijkstra(adj_matrix, start):
    n = len(adj_matrix)
    distances = [float('inf')] * n
    distances[start] = 0
    visited = [False] * n
    pq = [(0, start)]  # priority queue

    while pq:
        curr_dist, curr_vertex = heapq.heappop(pq)  # pop the vertex with the smallest distance

        if visited[curr_vertex]:
            continue

        visited[curr_vertex] = True

        for neighbor in range(n):
            if adj_matrix[curr_vertex][neighbor] != 0 and not visited[neighbor]:
                new_dist = curr_dist + adj_matrix[curr_vertex][neighbor]
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))

    for i in range(n):
        if distances[i] == float('inf'):
            distances[i] = 0
        distances[i] = int(distances[i])
    return distances


def calculate_depth(adj_matrix):
    n = len(adj_matrix)
    depth = [0] * n
    visited = [False] * n
    # use_visited = False

    # if has_cycle(adj_matrix):
    #     use_visited = True
    use_visited = True
    def bfs(start, str, use_visited):
        queue = [start]
        if str == "root":
            depth[start] = 1
        else:
            have_child = False
            for neighbor in range(n):
                if adj_matrix[start][neighbor] == 1:
                    have_child = True
                    break
            
            if have_child:
                depth[start] = 1
            else:
                depth[start] = 0
                return
                
        visited[start] = True
        while queue:
            current = queue.pop(0)
            for neighbor in range(n):
                if use_visited:
                    if adj_matrix[current][neighbor] == 1 and not visited[neighbor]:
                        queue.append(neighbor)
                        depth[neighbor] = max(depth[current] + 1, depth[neighbor])
                        visited[neighbor] = True
                else:
                    if adj_matrix[current][neighbor] == 1:
                        queue.append(neighbor)
                        depth[neighbor] = max(depth[current] + 1, depth[neighbor])
    
    bfs(0, "root", use_visited)
    if len(visited) > 1 and not any(visited[1:]):
        input_str = "root"
        for i in range(n):
            if not visited[i]:
                bfs(i, input_str, use_visited)
    # else:
    #     input_str = "not_root"
    #     for i in range(n):
    #         if not visited[i]:
    #             bfs(i, input_str, use_visited)
    
    return depth


def has_cycle(adj_matrix):
    n = len(adj_matrix)
    in_degree = [0] * n
    topological_order = []
    
    # in_degree of each node
    for i in range(n):
        for j in range(n):
            if adj_matrix[j][i] == 1:
                in_degree[i] += 1
    
    # find the nodes with in_degree 0
    queue = [i for i in range(n) if in_degree[i] == 0]
    
    # topological sort
    while queue:
        current = queue.pop(0)
        topological_order.append(current)
        for neighbor in range(n):
            if adj_matrix[current][neighbor] == 1:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
    
    # check if there is a cycle
    return len(topological_order) != n


class MLPforBiaffine(nn.Module):
    def __init__(self, input_dim, concat_dim, d_inner):
        super(MLPforBiaffine, self).__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, d_inner),
            nn.ReLU(),
            nn.Linear(d_inner, 2*input_dim)
        )

        nn.init.kaiming_normal_(self.mlp[1].weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.mlp[1].bias)
        self.relu = nn.ReLU()
    
    def forward(self, inp):
        x = self.mlp(inp)
        return x

class BottleNeck(nn.Module):
    def __init__(self, d_model, concat_dim, d_inner):
        super(BottleNeck, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.L_1 = nn.Linear(concat_dim, self.d_model)
        self.relu = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, d_inner),
            nn.ELU(),
            nn.Linear(d_inner, self.d_model)
        )
        self.mlp2 = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, d_inner),
            nn.ELU(),
            nn.Linear(d_inner, 2048)
        )
        nn.init.kaiming_normal_(self.L_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.L_1.bias)
        self.proj = nn.Linear(2048, 256, bias=False)
    
    def forward(self, x):
        x = self.relu(self.L_1(x))
        x = self.mlp(x)
        x = self.mlp2(x)
        return self.proj(x)
        # return x

class BiaffineAttention(nn.Module):
    def __init__(self, d_model, input_dim, embed_len, type="default"):
        super(BiaffineAttention, self).__init__()
        self.input_dim = input_dim
        # self.depth_embed = nn.ModuleList([torch.nn.Embedding(151, embed_len[0]),
        #         torch.nn.Embedding(151, embed_len[1]),
        #         torch.nn.Embedding(151, embed_len[2]),
        #         torch.nn.Embedding(151, embed_len[3])])
        self.depth_embed = nn.ModuleList([torch.nn.Embedding(151, 1024),
                torch.nn.Embedding(151, 1024),
                torch.nn.Embedding(151, 1024),
                torch.nn.Embedding(151, 1024)])
        self.dep_to_parent_k = nn.Linear(4 * input_dim, 2 * input_dim, bias=False)
        self.dep_to_child_k = nn.Linear(4 * input_dim, 2 * input_dim, bias=False)

        self.type = type
        self.temperature = 1.0
        self.concat_dim = input_dim * 3

        self.f_1 = MLPforBiaffine(input_dim, self.concat_dim, self.concat_dim)
        self.f_3 = MLPforBiaffine(input_dim, self.concat_dim, self.concat_dim)
        self.b_1 = nn.Parameter(torch.rand(input_dim + 1, input_dim + 1))

        self.f_2 = MLPforBiaffine(input_dim // 2, input_dim * 2, input_dim * 2)
        self.f_4 = MLPforBiaffine(input_dim // 2, input_dim * 2, input_dim * 2)

        self.pos_emb = PositionalEmbedding(input_dim)
        self.pos_to_k = nn.Linear(input_dim, 2 * input_dim, bias=False)
        
        if type == "default":
            self.softmax = nn.Softmax(dim=-1)
        elif type == "Multi":
            self.root_representation = nn.Parameter(torch.Tensor(1, input_dim * 3))
            self.softmax = nn.Sigmoid()
            nn.init.xavier_uniform_(self.root_representation)
        
    def forward(self, hidden, attn_relpos): # i*d -> j*i (j = i + 1)
        batchsize = hidden.shape[0]
        child_hidden = self.f_3(hidden) # b*i*d
        if self.type == "Multi":
            repeat_shape = list(hidden.shape)
            repeat_shape[-1] = 1
            repeat_shape[-2] = 1
            root_tokens = self.root_representation.repeat(*repeat_shape)
            hidden = torch.cat((root_tokens, hidden), dim=hidden.dim() - 2)
        parent_hidden = self.f_1(hidden)
        time_step = parent_hidden.shape[-2] - 1 # time vary position embedding
        parent_hidden = parent_hidden.unsqueeze(1).repeat(1, time_step, 1, 1)    # b*i*j*d
        child_hidden = child_hidden.unsqueeze(1).repeat(1, time_step + 1, 1, 1).permute(0, 2, 1, 3)
        # scores = torch.matmul(parent_hidden, child_hidden.transpose(-1, -2))
        attn_relpos = torch.clip(attn_relpos, 0, 150).long()
        child_relpos = attn_relpos.clone()
        for i in range(len(attn_relpos)): 
            child_relpos[i, :] = F.pad(child_relpos[i, :].permute(0, 2, 1)[:, 1:, :], (1, 0))   # b*i*j
        embed_tuple_parent = [self.depth_embed[i](attn_relpos[i]) for i in range(len(attn_relpos))]
        embed_tuple_child = [self.depth_embed[i](child_relpos[i]) for i in range(len(child_relpos))]
        embed_biases_parent = torch.cat(embed_tuple_parent, dim=-1)   # b*i*j*d
        embed_biases_child = torch.cat(embed_tuple_child, dim=-1)
        embed_biases_parent = self.dep_to_parent_k(embed_biases_parent)
        embed_biases_child = self.dep_to_child_k(embed_biases_child)
        
        position = torch.arange(hidden.size(1), device=hidden.device) - torch.arange(hidden.size(1), device=hidden.device).unsqueeze(1)[1:, :]
        position = (torch.stack([self.pos_emb(pos) for pos in position])).permute(2, 0, 1, 3).repeat(batchsize, 1, 1, 1)
        pos_info = self.pos_to_k(position)
        parent_hidden = parent_hidden + pos_info + embed_biases_parent
        child_hidden = child_hidden + embed_biases_child

        parent_hidden = self.f_2(parent_hidden)
        child_hidden = self.f_4(child_hidden)

        scores = torch.einsum('bijd,dd,bijd->bji', F.pad(parent_hidden, (0, 1)), self.b_1, F.pad(child_hidden, (0, 1)))
        scores = self.softmax(scores)
        return scores
    
    def set_temperature(self, value):
        self.temperature = value

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
        # embed_len = kwargs.pop('embed_len', None)
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        self.embed_k_net = nn.ModuleList([torch.nn.Linear(self.d_model, self.d_model) for i in range(4)])
        # if embed_len[0] is not None:
        # self.rel_embed = nn.ModuleList([torch.nn.Embedding(151, 1024),
        #     torch.nn.Embedding(151, 1024),
        #     torch.nn.Embedding(151, 1024),
        #     torch.nn.Embedding(151, 1024)])
        #     self.pos_rel_embed = nn.ModuleList([torch.nn.Embedding(151, embed_len[0]),
        #         torch.nn.Embedding(151, embed_len[1]),
        #         torch.nn.Embedding(151, embed_len[2]),
        #         torch.nn.Embedding(151, embed_len[3])])
        # else:
        #     self.content_rel_embed = torch.nn.Embedding(151, self.n_head * self.d_head)
        #     self.pos_rel_embed = torch.nn.Embedding(151, self.n_head * self.d_head)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, attn_relpos=None, min_len=None, max_len=None
        , mems=None, terminal=False, past_keys=None, past_values=None, cache=False, content_rel_embed=None):
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
            if content_rel_embed is not None:
                r2 = torch.arange(max_len - min_len, -1, -1.0, device=w.device).long()
                W_k = self.qkv_net[0].weight[self.d_model:2 * self.d_model, :]
                # R_k = self.r_net.weight
                biases = [rel_embed(r2) for rel_embed in content_rel_embed]
                biases = [self.embed_k_net[i](biases[i]) for i in range(4)]
                rel_wk = [(bias@W_k.T).view(max_len - min_len + 1, self.n_head, self.d_head) for bias in biases]
                # if attn_relpos.dim() == 4:
                #     # content_rel_embed, pos_rel_embed = content_rel_embed
                #     degree_embed, distance_embed, depth_embed, predepth_embed = content_rel_embed
                #     rel_len = [embed_layer.weight.shape[1] for embed_layer in content_rel_embed]
                #     rel_wk = []
                #     # rel_rk = []
                #     start_pos, end_pos = 0, rel_len[0]
                #     for i in range(attn_relpos.dim()):
                #         biases = content_rel_embed[i](r2)
                #         # biases2 = pos_rel_embed[i](r2)
                #         rel_wk.append((biases @ W_k[:, start_pos:end_pos].T).view(max_len - min_len + 1, self.n_head, self.d_head))
                #         # rel_rk.append((biases2 @ R_k[:, start_pos:end_pos].T).view(max_len - min_len + 1, self.n_head, self.d_head))
                #         start_pos += rel_len[i]
                #         if i != attn_relpos.dim() - 1:
                #             end_pos += rel_len[i + 1]
                # else:
                #     depth_biases = content_rel_embed(r2)
                #     # Moderate_rk = self.r_net(depth_biases)
                #     Moderate_wk = depth_biases @ W_k.T
                #     Moderate_wk = Moderate_wk.view(max_len - min_len + 1, self.n_head, self.d_head)
                #     # Moderate_rk = Moderate_rk.view(max_len - min_len + 1, self.n_head, self.d_head)

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
        if attn_relpos is None:
            BD = self._rel_shift(BD)
        else:
            # RkD
            if attn_relpos.dim() != 4:
                attn_relpos = torch.clip(attn_relpos, min_len, max_len).long()
                # attn_relpos = torch.randint(min_len, max_len, size=attn_relpos.shape).long().cuda()
                # attn_relpos = torch.zeros_like(attn_relpos)
                # attn_relpos = torch.arange(1, attn_relpos.shape[1] + 1).unsqueeze(0).repeat(attn_relpos.shape[0], attn_relpos.shape[2], 1).cuda()
                attn_relpos = attn_relpos.permute(1, 2, 0)
                # Moderate_BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, Moderate_rk))
                Moderate_AC = torch.einsum('ibnd,jnd->ijbn', (rw_head_q, Moderate_wk))
                attn_relpos = (max_len - attn_relpos).long()
                # BD = self._rel_shift(BD) + Moderate_BD.gather(1, attn_relpos.unsqueeze(-1).expand(-1, -1, -1, BD.shape[-1])) + Moderate_AC.gather(1, attn_relpos.unsqueeze(-1).expand(-1, -1, -1, BD.shape[-1]))
                BD = self._rel_shift(BD) + Moderate_AC.gather(1, attn_relpos.unsqueeze(-1).expand(-1, -1, -1, BD.shape[-1]))
                # BD.gather(1, attn_relpos.unsqueeze(-1).expand(-1, -1, -1, BD.shape[-1]))
            else:
                # min_len = -75
                attn_relpos = torch.clip(attn_relpos, min_len, max_len).long()
                attn_relpos = attn_relpos.permute(0, 2, 3, 1)
                attn_relpos = (max_len - attn_relpos).long()
                # depth_relpos = (depth_relpos - 75).long()
                # depth_relpos = torch.clip(depth_relpos, min_len, max_len).long()
                Moderate_AC_rel = [torch.einsum('ibnd,jnd->ijbn', (rw_head_q, item)) for item in rel_wk]   # exchange position
                BD = self._rel_shift(BD)
                for idx, item in enumerate(Moderate_AC_rel):
                    BD = BD + item.gather(1, attn_relpos[idx].unsqueeze(-1).expand(-1, -1, -1, BD.shape[-1]))
                # Moderate_BD_rel = [torch.einsum('ibnd,jnd->ijbn', (rr_head_q, item)) for item in rel_rk]
                # for idx, item in enumerate(Moderate_BD_rel):
                #     BD = BD + item.gather(1, attn_relpos[idx].unsqueeze(-1).expand(-1, -1, -1, BD.shape[-1]))

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
    def forward(self, dec_inp, r, r_w_bias, r_r_bias, attn_mask=None, attn_relpos=None, min_len=None, max_len=None,
        mems=None, terminal=False, past_keys=None, past_values=None, cache=False, rel_embed=None):
        if cache:
            output, new_key, new_value = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                                attn_mask=attn_mask, attn_relpos=attn_relpos,
                                min_len=min_len, max_len=max_len, mems=mems,
                                terminal=terminal, past_keys=past_keys, past_values=past_values,
                                cache=cache, content_rel_embed=rel_embed)
            output = self.pos_ff(output)

            return output, new_key, new_value
        else:
            output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                                attn_mask=attn_mask, attn_relpos=attn_relpos,
                                min_len=min_len, max_len=max_len, mems=mems, terminal=terminal,
                                past_keys=past_keys, past_values=past_values, content_rel_embed=rel_embed)
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
                 pre_lnorm = False,
                 rel_type = None,
                 degree_len = None,
                 depth_len = None,
                 distance_len = None,
                 predepth_len = None):
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
        self.rel_type = rel_type
        if rel_type == "mixing":
            self.degree_len = degree_len
            self.distance_len = distance_len
            self.depth_len = depth_len
            self.predepth_len = predepth_len
            # self.rel_embed = nn.ModuleList([torch.nn.Embedding(151, degree_len),
            #     torch.nn.Embedding(151, depth_len),
            #     torch.nn.Embedding(151, distance_len),
            #     torch.nn.Embedding(151, predepth_len)])
            self.rel_embed = nn.ModuleList([torch.nn.Embedding(151, 1024),
                torch.nn.Embedding(151, 1024),
                torch.nn.Embedding(151, 1024),
                torch.nn.Embedding(151, 1024)])
            assert self.n_head * self.d_head == degree_len + distance_len + depth_len + predepth_len
        else:
            self.rel_embed = None
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
        max_relative_length=None, min_relative_length=None, iseval=False, sents_index_to_id=None,
        sents_arrow=None, rel_type='degree', finetune=None):
        
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
            attn_relpos_for_pointer = []
            if rel_type == "degree" or rel_type == "mixing":
                for left_sent_arrow, right_sent_arrow, sent_index_to_id in zip(left_sents_arrow, right_sents_arrow, sents_index_to_id):
                    input_size = len(sent_index_to_id) - 1
                    id_to_index = {}
                    for i in range(len(sent_index_to_id)):
                        if sent_index_to_id[i] != -1:
                            if sent_index_to_id[i] not in id_to_index:
                                id_to_index[sent_index_to_id[i]] = [i]
                            else:
                                id_to_index[sent_index_to_id[i]].append(i)

                    sent_attn_relpos = np.zeros((length_i, length_i))
                    sent_attn_relpos_step = np.zeros((length_i))
                    sent_attn_relpos_for_pointer = np.zeros((max(sent_index_to_id) + 1, (max(sent_index_to_id)) + 1))
                    finished_word_idx = -1
                    degree_for_pointer = np.zeros((max(sent_index_to_id) + 1))
                    for i in range(input_size):
                        if sent_index_to_id[i] == -1:
                            if finetune is not None:
                                sent_attn_relpos_step = np.zeros((length_i))
                                degree_for_pointer = np.zeros((max(sent_index_to_id) + 1))
                            continue
                        tmp_finished_word_idx = finished_word_idx
                        finished_word_idx = sent_index_to_id[i] # first token i - 1
                        if finished_word_idx != tmp_finished_word_idx:
                            add_arc_idx = finished_word_idx
                            if left_sent_arrow[add_arc_idx - 1]:  #i <- j
                                for j in left_sent_arrow[add_arc_idx - 1]:
                                    if j != 0:
                                        sent_attn_relpos_step[id_to_index[j]] += 10 # 10
                                    sent_attn_relpos_step[id_to_index[add_arc_idx]] += 1
                                    degree_for_pointer[j] += 10
                                    degree_for_pointer[add_arc_idx] += 1

                            if right_sent_arrow[add_arc_idx - 1]: #i -> j
                                for j in right_sent_arrow[add_arc_idx - 1]:
                                    if j != 0:
                                        sent_attn_relpos_step[id_to_index[j]] += 1
                                    sent_attn_relpos_step[id_to_index[add_arc_idx]] += 10 # 10
                                    degree_for_pointer[j] += 1
                                    degree_for_pointer[add_arc_idx] += 10
                        sent_attn_relpos[i] = copy.deepcopy(sent_attn_relpos_step)
                        sent_attn_relpos_for_pointer[add_arc_idx] = copy.deepcopy(degree_for_pointer)
                        # Wk R
                        # rel_pos_1 = rel_pos_1[(i+1):] + rel_pos_1[:(i+1)]
                        # rel_pos_2 = rel_pos_2[(i+1):] + rel_pos_2[:(i+1)]
                    # for j in range(length_i - len(sent_attn_relpos)):
                    #     sent_attn_relpos.append([0]*(length_i))
                    attn_relpos.append(sent_attn_relpos)
                    attn_relpos_for_pointer.append(sent_attn_relpos_for_pointer[:-1])
                    # attn_relpos.append(10 * np.array(Degree_out[:-1]) + 1 * np.array(Degree_in[:-1]))
            if rel_type == "depth" or rel_type == "rel_depth" or rel_type == "mixing":
                for left_sent_arrow, right_sent_arrow, sent_index_to_id in zip(left_sents_arrow, right_sents_arrow, sents_index_to_id):
                    input_size = len(sent_index_to_id) - 1
                    id_to_index = {}
                    for i in range(len(sent_index_to_id)):
                        if sent_index_to_id[i] != -1:
                            if sent_index_to_id[i] not in id_to_index:
                                id_to_index[sent_index_to_id[i]] = [i]
                            else:
                                id_to_index[sent_index_to_id[i]].append(i)
                    
                    graph_len = len(left_sent_arrow) + 1
                    graph = np.zeros((graph_len, graph_len))
                    sent_attn_relpos = np.zeros((length_i, length_i))
                    sent_attn_relpos_for_pointer = np.zeros((max(sent_index_to_id) + 1, (max(sent_index_to_id)) + 1))
                    sent_attn_relpos_for_pointer[0, 0] = 1  # root
                    for i in range(len(left_sent_arrow)):
                        if left_sent_arrow[i]:  #i <- j
                            for j in left_sent_arrow[i]:
                                graph[j][i + 1] = 1
                                graph[i + 1][j] = 1
                        if right_sent_arrow[i]: #i -> j
                            for j in right_sent_arrow[i]:
                                graph[i + 1][j] = 1
                                graph[j][i + 1] = 1
                    
                    last_finished_word_idx = None
                    finished_word_idx = None
                    for i in range(input_size):
                        if sent_index_to_id[i] == -1:
                            if finetune is not None:
                                last_finished_word_idx = finished_word_idx
                            continue
                        finished_word_idx = sent_index_to_id[i] # first token max(sent_index_to_id[i - 1], 0)
                        depth = calculate_depth(graph[:finished_word_idx + 1, :finished_word_idx + 1])
                        if last_finished_word_idx:
                            depth[1:last_finished_word_idx + 1] = [0] * last_finished_word_idx
                        # if rel_type == "rel_depth":
                        # depth = [item - depth[finished_word_idx] for item in depth]
                        for id, depth_value in enumerate(depth[1:]):
                            sent_attn_relpos[i, id_to_index[id + 1]] = depth_value
                        for id, depth_value in enumerate(depth):
                            sent_attn_relpos_for_pointer[finished_word_idx, id] = depth_value
                    # for j in range(length_i - len(sent_attn_relpos)):
                    #     sent_attn_relpos.append([0]*(length_i))
                    attn_relpos_for_pointer.append(sent_attn_relpos_for_pointer[:-1])
                    attn_relpos.append(sent_attn_relpos)
            if rel_type == "distance" or rel_type == "mixing":
                for left_sent_arrow, right_sent_arrow, sent_index_to_id in zip(left_sents_arrow, right_sents_arrow, sents_index_to_id):
                    input_size = len(sent_index_to_id) - 1
                    id_to_index = {}
                    for i in range(len(sent_index_to_id)):
                        if sent_index_to_id[i] != -1:
                            if sent_index_to_id[i] not in id_to_index:
                                id_to_index[sent_index_to_id[i]] = [i]
                            else:
                                id_to_index[sent_index_to_id[i]].append(i)
                    
                    graph_len = len(left_sent_arrow) + 1
                    graph = np.zeros((graph_len, graph_len))
                    sent_attn_relpos = np.zeros((length_i, length_i))
                    sent_attn_relpos_for_pointer = np.zeros((max(sent_index_to_id) + 1, (max(sent_index_to_id)) + 1))
                    for i in range(len(left_sent_arrow)):
                        if left_sent_arrow[i]:  #i <- j
                            for j in left_sent_arrow[i]:
                                graph[j][i + 1] = 10
                                graph[i + 1][j] = 1  #2
                        if right_sent_arrow[i]: #i -> j
                            for j in right_sent_arrow[i]:
                                graph[i + 1][j] = 10
                                graph[j][i + 1] = 1  #2

                    last_finished_word_idx = None
                    finished_word_idx = None
                    for i in range(input_size):
                        if sent_index_to_id[i] == -1:
                            if finetune is not None:
                                last_finished_word_idx = finished_word_idx
                            continue
                        finished_word_idx = sent_index_to_id[i] # first token max(sent_index_to_id[i - 1], 0)
                        distance = dijkstra(graph[:finished_word_idx + 1, :finished_word_idx + 1], finished_word_idx)
                        if last_finished_word_idx:
                            distance[1:last_finished_word_idx + 1] = [0] * last_finished_word_idx
                        for id, distance_value in enumerate(distance[1:]):
                            sent_attn_relpos[i, id_to_index[id + 1]] = distance_value
                        for id, distance_value in enumerate(distance):
                            sent_attn_relpos_for_pointer[finished_word_idx, id] = distance_value
                    # for j in range(length_i - len(sent_attn_relpos)):
                        # sent_attn_relpos.append([0]*(length_i))
                    attn_relpos_for_pointer.append(sent_attn_relpos_for_pointer[:-1])
                    attn_relpos.append(sent_attn_relpos)
            if rel_type == "predicate_depth" or rel_type == "mixing":
                for left_sent_arrow, right_sent_arrow, sent_index_to_id in zip(left_sents_arrow, right_sents_arrow, sents_index_to_id):
                    input_size = len(sent_index_to_id) - 1
                    id_to_index = {}
                    for i in range(len(sent_index_to_id)):
                        if sent_index_to_id[i] != -1:
                            if sent_index_to_id[i] not in id_to_index:
                                id_to_index[sent_index_to_id[i]] = [i]
                            else:
                                id_to_index[sent_index_to_id[i]].append(i)
                    
                    sent_attn_relpos = np.zeros((length_i, length_i))
                    sent_attn_relpos_for_pointer = np.zeros((max(sent_index_to_id) + 1, (max(sent_index_to_id)) + 1))
                    father_tag = np.zeros(len(left_sent_arrow))
                    finished_word_idx = -1
                    for i in range(input_size):
                        if sent_index_to_id[i] == -1:
                            if finetune is not None:
                                father_tag = np.zeros(len(left_sent_arrow))
                            continue
                        tmp_finished_word_idx = finished_word_idx
                        finished_word_idx = sent_index_to_id[i] # first token i - 1
                        if finished_word_idx != tmp_finished_word_idx:
                            if right_sent_arrow[finished_word_idx - 1]:  #i <- j
                                father_tag[[idx - 1 for idx in right_sent_arrow[finished_word_idx - 1] if idx != 0]] = 1
                            if left_sent_arrow[finished_word_idx - 1]: #i -> j
                                father_tag[finished_word_idx - 1] = 1
                        depth = 0
                        for idx in range(finished_word_idx, 0, -1):
                            if not father_tag[idx - 1]:
                                depth += 1
                                sent_attn_relpos[i, id_to_index[idx]] = depth
                                sent_attn_relpos_for_pointer[finished_word_idx, idx] = depth
                            else:
                                sent_attn_relpos[i, id_to_index[idx]] = 0
                                sent_attn_relpos_for_pointer[finished_word_idx, idx] = 0
                    attn_relpos.append(sent_attn_relpos) 
                    attn_relpos_for_pointer.append(sent_attn_relpos_for_pointer[:-1])
            
            attn_relpos = torch.LongTensor(np.array(attn_relpos))
            if rel_type == "mixing":
                attn_relpos = attn_relpos.reshape(4, batch, length_i, length_i)
                attn_relpos_for_pointer = [torch.LongTensor(np.array([attn_relpos_for_pointer[i + j * batch] for j in range(4)])) for i in range(batch)]
            if torch.cuda.is_available():
                attn_relpos = attn_relpos.cuda()
                attn_relpos_for_pointer = [attn_relpos_for_pointer[i].cuda() for i in range(batch)]

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
            min_relative_length = 0
            if use_mask == 'graphlayer':
                max_relative_length = 150
                if rel_type == "rel_depth":
                    max_relative_length = 75
                    min_relative_length = -75
            else:
                max_relative_length = seq_len - 1
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
                attn_relpos=attn_relpos, min_len=min_relative_length, max_len=max_relative_length,
                rel_embed = self.rel_embed)
            hiddens.append(core_out)
            if i < len(self.layers) - 1:
                core_out = self.dropout(core_out)
        core_out = self.dropout(core_out)

        logits = self.projection(core_out) 
        crit = nn.CrossEntropyLoss(reduction='none', ignore_index=self.pad_id)
        prob = logits.view(seq_len, batch, -1)
        prob = prob.permute(0, 2, 1)
        loss = crit(prob, targets)
        if finetune is not None:
            mask = torch.zeros_like(loss)
            for j, sent in enumerate(x):
                if finetune == "sst2":
                    idx = find_label_idx(sent, [26858, 5599]) + 1
                elif finetune == "mrpc" or finetune == "rte":
                    idx = find_label_idx(sent, [2745, 11346, 5599]) + 2
                mask[idx, j] = 1
            loss = loss * mask
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
            if use_mask == "graphlayer":
                return loss, torch.cat([hiddens[9], hiddens[15]], dim = -1), word_emb, attn_relpos_for_pointer, prob
                # return loss, hiddens[9] + hiddens[15]
            return loss, core_out, prob
        else:
            return loss, prob

    
    def GraphlayerLM_inference(self, x, past_keys, past_values, attn_relpos = None):
        with torch.no_grad():
            inp = x.permute(1, 0).contiguous()  # 1 * batch new token
            word_emb = self.emb(inp)
            seq_len = attn_relpos.shape[3]
            pos_emb = self.pos_emb(torch.arange(seq_len - 1, -1, -1.0, device = word_emb.device))
            batch = x.shape[0]
            new_keys = torch.full((self.num_layers, 1, batch, self.w_dim), 0.0, device=word_emb.device)
            new_values = torch.full((self.num_layers, 1, batch, self.w_dim), 0.0, device=word_emb.device)
            
            past_keys_p = None
            past_values_p = None
            if past_keys is not None:
                past_keys_p = past_keys.reshape(past_keys.size(0), past_keys.size(1), self.num_layers, self.w_dim)
                past_values_p = past_values.reshape(past_values.size(0), past_values.size(1), self.num_layers, self.w_dim)
                past_keys_p = past_keys_p.permute(2, 1, 0, 3).contiguous() # layer * L-1 * batch * w_dim
                past_values_p = past_values_p.permute(2, 1, 0, 3).contiguous() 

            hiddens = []
            core_out = word_emb
            hiddens.append(core_out)
            for i, layer in enumerate(self.layers):
                if past_values_p is not None:
                    core_out, new_key, new_value = \
                                    layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, 
                                    attn_mask=None, attn_relpos=attn_relpos, 
                                    min_len=0, max_len=150, rel_embed = self.rel_embed,
                                    past_keys=past_keys_p[i], past_values=past_values_p[i], cache=True)
                else:
                    core_out, new_key, new_value = \
                                    layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, 
                                    attn_mask=None, attn_relpos=attn_relpos, 
                                    min_len=0, max_len=150, rel_embed = self.rel_embed,
                                    past_keys=None, past_values=None, cache=True)
                hiddens.append(core_out)
                new_keys[i] = new_key
                new_values[i] = new_value
            
            logits = self.projection(core_out) 
            prob = logits.view(1, batch, -1)
            prob = prob.log_softmax(-1)
            new_keys = new_keys.permute(2, 1, 0, 3).contiguous()
            new_values = new_values.permute(2, 1, 0, 3).contiguous()
            new_keys = new_keys.reshape(new_keys.size(0), new_keys.size(1), -1)
            new_values = new_values.reshape(new_values.size(0), new_values.size(1), -1)

            return prob, new_keys, new_values, torch.cat([hiddens[9], hiddens[15]], dim = -1)
    

    def get_emb(self, input_id):
        return self.emb(input_id)


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

                