import copy
from collections import deque, defaultdict
import heapq
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import time
import os
from helping_utils.logger import configure_logger, get_logger
logger = get_logger()

mixing_num = 2 # ablation

def find_label_idx(sent, prefix):
    main_str = ",".join(map(str, sent))
    sub_str = ",".join(map(str, prefix))
    idx = main_str.find(sub_str)
    if idx == -1:
        return -1
    return main_str[:idx].count(",") if idx != 0 else 0

class IncrementalBellmanFord:
    def __init__(self, n):
        self.n = n
        self.adj = defaultdict(list)
        self.dist = [float('inf')] * n
        self.in_queue = [False] * n

    def add_edge(self, u, v, w):
        self.adj[u].append((v, w))
        self._update(u)

    def _update(self, start):
        q = deque()
        q.append(start)
        self.in_queue[start] = True
        while q:
            u = q.popleft()
            self.in_queue[u] = False
            for v, w in self.adj[u]:
                if self.dist[u] + w < self.dist[v]:
                    self.dist[v] = self.dist[u] + w
                    if not self.in_queue[v]:
                        q.append(v)
                        self.in_queue[v] = True

    def set_source(self, src):
        self.dist = [float('inf')] * self.n
        self.dist[src] = 0
        self._update(src)

    def get_dist_from_v(self, v):
        return self.dist[v] if self.dist[v] != float('inf') else None
    
    def get_dist(self):
        return self.dist
    
    def get_depth_for_GiLT(self):
        return [item + 1 if item != float('inf') else 0 for item in self.dist]
    
    def get_dist_for_GiLT(self):
        return [item if item != float('inf') else 0 for item in self.dist]


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

def dijkstra_heap(mat, src):
    """O(m log n) heapq 版，把邻接矩阵当成稀疏图用"""
    n = len(mat)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            w = mat[i][j]
            if w:
                adj[i].append((j, w))
    dist = [float('inf')] * n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    for i in range(n):
        if dist[i] == float('inf'):
            dist[i] = 0
        dist[i] = int(dist[i])
    return dist


def calculate_depth(adj_matrix):
    # n = len(adj_matrix)
    # depth = [0] * n
    # visited = [False] * n
    # use_visited = True
    # def bfs(start, str, use_visited):
    #     queue = [start]
    #     if str == "root":
    #         depth[start] = 1
    #     else:
    #         have_child = False
    #         for neighbor in range(n):
    #             if adj_matrix[start][neighbor] == 1:
    #                 have_child = True
    #                 break
            
    #         if have_child:
    #             depth[start] = 1
    #         else:
    #             depth[start] = 0
    #             return
                
    #     visited[start] = True
    #     while queue:
    #         current = queue.pop(0)
    #         for neighbor in range(n):
    #             if use_visited:
    #                 if adj_matrix[current][neighbor] == 1 and not visited[neighbor]:
    #                     queue.append(neighbor)
    #                     depth[neighbor] = max(depth[current] + 1, depth[neighbor])
    #                     visited[neighbor] = True
    #             else:
    #                 if adj_matrix[current][neighbor] == 1:
    #                     queue.append(neighbor)
    #                     depth[neighbor] = max(depth[current] + 1, depth[neighbor])
    
    # bfs(0, "root", use_visited)

    # if len(visited) > 1 and not any(visited[1:]):
    #     input_str = "root"
    #     for i in range(n):
    #         if not visited[i]:
    #             bfs(i, input_str, use_visited)
    # else:
    #     input_str = "not_root"
    #     for i in range(n):
    #         if not visited[i]:
    #             bfs(i, input_str, use_visited)
    depth = dijkstra_heap(adj_matrix, 0)
    depth = [value+1 if value != 0 else value for value in depth]
    depth[0] = 1
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
    def __init__(self, output_dim, input_dim, d_inner):
        super(MLPforBiaffine, self).__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_inner),
            nn.ReLU(),
            nn.Linear(d_inner, output_dim)
        )
        nn.init.kaiming_normal_(self.mlp[1].weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.mlp[1].bias)
        nn.init.kaiming_normal_(self.mlp[3].weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.mlp[3].bias)
    
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
    def __init__(self, out_dim, input_dim, n_head = 1, type="default"):
        super(BiaffineAttention, self).__init__()
        self.input_dim = input_dim
        self.type = type
        self.temperature = 1.0
        self.concat_dim = input_dim * 3
        self.n_head = n_head
        self.head_dim = out_dim
        self.mlp1_in_dim = input_dim * 3
        self.mlp1_out_dim = 2048

        self.f_1 = MLPforBiaffine(self.mlp1_out_dim, self.mlp1_in_dim, 3072)  #3072->3072->2048
        self.f_3 = MLPforBiaffine(self.mlp1_out_dim, self.mlp1_in_dim, 3072)
        self.f_2 = MLPforBiaffine(1024, self.mlp1_out_dim, 2048)  #2048->2048->1024
        self.f_4 = MLPforBiaffine(1024, self.mlp1_out_dim, 2048)
        # self.f_2 = nn.Sequential(MLPforBiaffine(out_dim // self.n_head, 2048 // self.n_head, 4 * input_dim // self.n_head)
        #     , MLPforBiaffine(out_dim // self.n_head, 2048 // self.n_head, 4 * input_dim // self.n_head))
        # self.f_4 = nn.Sequential(MLPforBiaffine(out_dim // self.n_head, 2048 // self.n_head, 4 * input_dim // self.n_head)
        #     , MLPforBiaffine(out_dim // self.n_head, 2048 // self.n_head, 4 * input_dim // self.n_head))

        self.pos_emb = PositionalEmbedding(1024)
        self.pos_to_parent_k = nn.Linear(1024, self.mlp1_out_dim, bias=False)
        self.pos_to_child_k = nn.Linear(1024, self.mlp1_out_dim, bias=False)
        self.depth_embed = nn.ModuleList([torch.nn.Embedding(151, 256) for _ in range(mixing_num)])
        self.dep_to_parent_k = nn.Linear(mixing_num * 256, self.mlp1_out_dim, bias=False)
        self.dep_to_child_k = nn.Linear(mixing_num * 256, self.mlp1_out_dim, bias=False)

        self.b_1 = nn.Parameter(torch.rand(self.head_dim, self.head_dim))
        self.b_2 = nn.Parameter(torch.rand(self.head_dim, self.head_dim))
        
        if type == "default":
            self.softmax = nn.Softmax(dim=-1)
        elif type == "Multi":
            self.root_representation = nn.Parameter(torch.Tensor(1, self.mlp1_in_dim))
            self.softmax = nn.Sigmoid()
        
        self.arc_softmax = nn.Softmax(dim=-1)
        self.prob = nn.Linear(self.head_dim, 30)
        if self.n_head != 1:
            self.onet = nn.Linear(self.n_head, 1)
        
        for layer in [self.pos_to_parent_k, self.pos_to_child_k, self.dep_to_child_k, self.dep_to_parent_k]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
        nn.init.xavier_normal_(self.root_representation)
        nn.init.kaiming_normal_(self.b_1, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.b_2, mode='fan_in', nonlinearity='linear')
        for embed_layer in self.depth_embed:
            fan_in = nn.init._calculate_correct_fan(embed_layer.weight, 'fan_in')
            nn.init.uniform_(embed_layer.weight, -np.sqrt(3 / fan_in), np.sqrt(3 / fan_in))
        
    def forward(self, hidden, attn_relpos): # i*d -> j*i (j = i + 1)
        # attn_relpos = torch.stack([attn_relpos[0], attn_relpos[1], attn_relpos[2]]).to(attn_relpos.device)
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
        
        position = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(1) - torch.arange(hidden.size(1), device=hidden.device)
        child_position = F.pad(torch.triu(position.T)[:, :-1], (1,0))[:-1, :]
        parent_position = torch.tril(position)[1:, :]
        parent_position = (torch.stack([self.pos_emb(pos) for pos in parent_position])).permute(2, 0, 1, 3).repeat(batchsize, 1, 1, 1)
        child_position = (torch.stack([self.pos_emb(pos) for pos in child_position])).permute(2, 0, 1, 3).repeat(batchsize, 1, 1, 1)
        parent_pos_info = self.pos_to_parent_k(parent_position)
        child_position_info = self.pos_to_child_k(child_position)
        parent_hidden = parent_hidden + parent_pos_info + embed_biases_parent
        child_hidden = child_hidden + child_position_info + embed_biases_child
        # parent_hidden = torch.cat((parent_hidden + parent_pos_info, embed_biases_parent), dim=-1)
        # child_hidden = torch.cat((child_hidden + child_position_info, embed_biases_child), dim=-1)

        parent_hidden = self.f_2(parent_hidden)
        child_hidden = self.f_4(child_hidden)

        if self.head_dim == 1024:
            logits = torch.einsum('bijd,dd,bijd->bji', parent_hidden, self.b_1[:1024,:1024], child_hidden)
            square = torch.einsum('bijn,nn,bijn->bijn', parent_hidden, self.b_2, child_hidden)
        else:
            square = torch.einsum('bijn,nn,bijn->bijn', F.pad(parent_hidden, (0, 1), value=1), self.b_2, F.pad(child_hidden, (0, 1), value=1))
            logits = torch.einsum('bijd,dd,bijd->bji', F.pad(parent_hidden, (0, 1), value=1), self.b_1, F.pad(child_hidden, (0, 1), value=1))
        steps = torch.arange(1, time_step + 1, device=hidden.device)
        scales = torch.sqrt(steps * 2).view(1, -1, 1)  # (1, time_step, 1)
        row_mask = torch.tril(torch.ones_like(position))[:-1, :]
        col_mask = torch.ones_like(position)[:-1, :] - row_mask
        row_sum = (square * row_mask.unsqueeze(-1)).sum(dim=-2)
        col_sum = (square * col_mask.unsqueeze(-1)).sum(dim=-3)[:, 1:, :]
        row_col_sum = row_sum + col_sum
        arc_num_scores = row_col_sum / scales   # b*l*n
        arc_num_logits = self.prob(arc_num_scores)
        arc_num_prob = self.arc_softmax(arc_num_logits)
        scores = self.softmax(logits)
        return scores, logits, arc_num_prob, arc_num_logits

    def multihead_forward(self, hidden, attn_relpos): # i*d -> j*i (j = i + 1)
        batchsize = hidden.shape[0]
        child_hidden = self.f_3(hidden) # b*i*d
        if self.type == "Multi":
            repeat_shape = [1]*len(hidden.shape)
            repeat_shape[0] = batchsize
            root_tokens = self.root_representation.repeat(*repeat_shape)
            # hidden = torch.cat((root_tokens, hidden), dim=1)
            parent_hidden = torch.cat((root_tokens, child_hidden), dim=1)
        # parent_hidden = self.f_1(hidden)
        time_step = parent_hidden.shape[1] - 1 # time vary position embedding
        parent_hidden = parent_hidden.unsqueeze(1).repeat(1, time_step, 1, 1)    # b*i*j*d
        child_hidden = child_hidden.unsqueeze(1).repeat(1, time_step + 1, 1, 1).permute(0, 2, 1, 3)
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
        position = torch.arange(time_step + 1, device=hidden.device).unsqueeze(1) - torch.arange(time_step + 1, device=hidden.device)
        child_position = F.pad(torch.triu(position.T)[:, :-1], (1,0))[:-1, :]
        parent_position = torch.tril(position)[1:, :]
        parent_position = (torch.stack([self.pos_emb(pos) for pos in parent_position])).permute(2, 0, 1, 3).repeat(batchsize, 1, 1, 1)
        child_position = (torch.stack([self.pos_emb(pos) for pos in child_position])).permute(2, 0, 1, 3).repeat(batchsize, 1, 1, 1)
        parent_pos_info = self.pos_to_parent_k(parent_position)
        child_position_info = self.pos_to_child_k(child_position)

        new_parent_shape = parent_hidden.shape[:-1] + (self.n_head, self.mlp1_out_dim // self.n_head)
        new_child_shape = child_hidden.shape[:-1] + (self.n_head, self.mlp1_out_dim // self.n_head)
        parent_hidden = parent_hidden.view(new_parent_shape)
        child_hidden = child_hidden.view(new_child_shape)

        parent_pos = (parent_pos_info + embed_biases_parent).unsqueeze(-2)
        parent_pos_head = parent_pos.expand(-1, -1, -1, self.n_head, -1) 
        parent_hidden = torch.cat((parent_hidden, parent_pos_head), dim=-1)
        child_pos = (child_position_info + embed_biases_child).unsqueeze(-2)
        child_pos_head = child_pos.expand(-1, -1, -1, self.n_head, -1) 
        child_hidden = torch.cat((child_hidden, child_pos_head), dim=-1)

        parent_hidden = self.f_2(parent_hidden)
        child_hidden = self.f_4(child_hidden)

        # square = torch.einsum('bijnd,dd,bijnd->bijd', F.pad(parent_hidden, (0, 1), value=1), self.b_2, F.pad(child_hidden, (0, 1), value=1))
        # logits = torch.einsum('bijnd,dd,bijnd->bjin', F.pad(parent_hidden, (0, 1), value=1), self.b_1, F.pad(child_hidden, (0, 1), value=1))
        square = torch.einsum('bijnd,dd,bijnd->bijd', parent_hidden, self.b_2, child_hidden)
        logits = torch.einsum('bijnd,dd,bijnd->bjin', parent_hidden, self.b_1, child_hidden)
        if self.n_head != 1:
            logits = self.onet(logits)
        logits = logits.squeeze(-1)
        steps = torch.arange(1, time_step + 1, device=hidden.device)
        scales = torch.sqrt(steps * 2).view(1, -1, 1)  # (1, time_step, 1)
        row_mask = torch.tril(torch.ones_like(position))[:-1, :]
        col_mask = torch.ones_like(position)[:-1, :] - row_mask
        row_sum = (square * row_mask.unsqueeze(-1)).sum(dim=-2)
        col_sum = (square * col_mask.unsqueeze(-1)).sum(dim=-3)[:, 1:, :]
        row_col_sum = row_sum + col_sum
        arc_num_scores = row_col_sum / scales   # b*l*n
        arc_num_logits = self.prob(arc_num_scores)
        arc_num_prob = self.arc_softmax(arc_num_logits)
        scores = self.softmax(logits)
        return scores, logits, arc_num_prob, arc_num_logits

    def set_temperature(self, value):
        self.temperature = value

    def inference(self, hidden, step_attn_relpos):
        batchsize = hidden.shape[0]
        child_hidden = self.f_3(hidden)
        if self.type == "Multi":
            repeat_shape = list(hidden.shape)
            repeat_shape[-1] = 1
            repeat_shape[-2] = 1
            root_tokens = self.root_representation.repeat(*repeat_shape)
            hidden = torch.cat((root_tokens, hidden), dim=hidden.dim() - 2)
        parent_hidden = self.f_1(hidden)

        step_attn_relpos = torch.clip(step_attn_relpos, 0, 150).long()
        child_relpos = F.pad(step_attn_relpos[:, :, 1:], (0,1))
        step_attn_relpos = F.pad(step_attn_relpos, (0,1))
        embed_tuple_parent = [self.depth_embed[i](step_attn_relpos[i]) for i in range(len(step_attn_relpos))]
        embed_tuple_child = [self.depth_embed[i](child_relpos[i]) for i in range(len(child_relpos))]
        embed_biases_parent = torch.cat(embed_tuple_parent, dim=-1)
        embed_biases_child = torch.cat(embed_tuple_child, dim=-1)
        embed_biases_parent = self.dep_to_parent_k(embed_biases_parent)
        embed_biases_child = self.dep_to_child_k(embed_biases_child)

        parent_position = torch.arange(child_hidden.size(1), -1, -1.0, device=hidden.device).long()
        parent_position = self.pos_emb(parent_position).permute(1, 0, 2).repeat(batchsize, 1, 1)
        child_position = parent_position[:,1:,:]
        parent_position_info = self.pos_to_parent_k(parent_position)
        child_position_info = self.pos_to_child_k(child_position)
        parent_hidden = parent_hidden + parent_position_info + embed_biases_parent
        child_hidden = child_hidden + child_position_info + embed_biases_child

        parent_hidden = self.f_2(parent_hidden)
        child_hidden = self.f_4(child_hidden)

        if self.head_dim == 1024:
            scores = torch.einsum('bjd,dd,bid->bji', parent_hidden, self.b_1[:1024,:1024], child_hidden)
            arc_num_scores = torch.einsum('bin, nn, bjn->bijn', parent_hidden, self.b_2, child_hidden)
        else:
            scores = torch.einsum('bjd,dd,bid->bji', F.pad(parent_hidden, (0, 1), value=1), self.b_1, F.pad(child_hidden, (0, 1), value=1))
            arc_num_scores = torch.einsum('bin, nn, bjn->bijn', F.pad(parent_hidden, (0, 1), value=1), self.b_2, F.pad(child_hidden, (0, 1), value=1))
        # scores = torch.zeros(batchsize, parent_hidden.shape[1], child_hidden.shape[1])
        # scores[:, -1:, :] = torch.einsum('bjd,dd,bid->bji', F.pad(parent_hidden[:,-1:,:], (0, 1)), self.b_1, F.pad(child_hidden, (0, 1)))
        # scores[:, :, -1:] = torch.einsum('bjd,dd,bid->bji', F.pad(parent_hidden, (0, 1)), self.b_1, F.pad(child_hidden[:,-1:,:], (0, 1)))
        scores = self.softmax(scores)
        # arc_num_scores = (torch.einsum('bn, nn, bjn->bjn', parent_hidden[:,-1,:], self.b_2, child_hidden[:,:-1,:]).sum(dim=-2) + torch.einsum('bin, nn, bn->bin', parent_hidden, self.b_2, child_hidden[:,-1,:]).sum(dim=-2)) / np.sqrt(2 * child_hidden.shape[1])
        # arc_num_scores = torch.einsum('bin, bjn->bijn', parent_hidden, child_hidden)
        arc_num_scores = (arc_num_scores[:, -1, :-1].sum(dim=-2) + arc_num_scores[:, :, -1].sum(dim=-2)) / np.sqrt(2 * arc_num_scores.shape[-2])
        arc_num_logits = self.prob(arc_num_scores)
        arc_num_prob = self.arc_softmax(arc_num_logits)
        return scores, arc_num_prob, arc_num_logits

    def multihead_inference(self, hidden, step_attn_relpos):
        batchsize = hidden.shape[0]
        child_hidden = self.f_3(hidden)
        if self.type == "Multi":
            repeat_shape = [1]*len(hidden.shape)
            repeat_shape[0] = batchsize
            root_tokens = self.root_representation.repeat(*repeat_shape)
            # hidden = torch.cat((root_tokens, hidden), dim=1)
            parent_hidden = torch.cat((root_tokens, child_hidden), dim=1)
        # parent_hidden = self.f_1(hidden)

        step_attn_relpos = torch.clip(step_attn_relpos, 0, 150).long()
        child_relpos = F.pad(step_attn_relpos[:, :, 1:], (0,1))
        step_attn_relpos = F.pad(step_attn_relpos, (0,1))
        embed_tuple_parent = [self.depth_embed[i](step_attn_relpos[i]) for i in range(len(step_attn_relpos))]
        embed_tuple_child = [self.depth_embed[i](child_relpos[i]) for i in range(len(child_relpos))]
        embed_biases_parent = torch.cat(embed_tuple_parent, dim=-1)
        embed_biases_child = torch.cat(embed_tuple_child, dim=-1)
        embed_biases_parent = self.dep_to_parent_k(embed_biases_parent)
        embed_biases_child = self.dep_to_child_k(embed_biases_child)

        parent_position = torch.arange(child_hidden.size(1), -1, -1.0, device=hidden.device).long()
        parent_position = self.pos_emb(parent_position).permute(1, 0, 2).repeat(batchsize, 1, 1)
        child_position = parent_position[:,1:,:]
        parent_position_info = self.pos_to_parent_k(parent_position)
        child_position_info = self.pos_to_child_k(child_position)
        # parent_hidden = parent_hidden + parent_position_info + embed_biases_parent
        # child_hidden = child_hidden + child_position_info + embed_biases_child

        new_shape_p = parent_hidden.shape[:-1] + (self.n_head, self.mlp1_out_dim // self.n_head)  # [B, L+1, n_head, head_dim]
        new_shape_c = child_hidden.shape[:-1]  + (self.n_head, self.mlp1_out_dim // self.n_head)  # [B, L,   n_head, head_dim]
        parent_hidden = parent_hidden.view(new_shape_p)
        child_hidden = child_hidden.view(new_shape_c)

        parent_pos = (parent_position_info + embed_biases_parent).unsqueeze(-2)
        parent_pos_head = parent_pos.expand(-1, -1, self.n_head, -1) 
        parent_hidden = torch.cat((parent_hidden, parent_pos_head), dim=-1)
        child_pos = (child_position_info + embed_biases_child).unsqueeze(-2)
        child_pos_head = child_pos.expand(-1, -1, self.n_head, -1) 
        child_hidden = torch.cat((child_hidden, child_pos_head), dim=-1)

        parent_hidden = self.f_2(parent_hidden)
        child_hidden = self.f_4(child_hidden)
        
        arc_num_scores = torch.einsum('bind, dd, bjnd->bijd', parent_hidden, self.b_2, child_hidden)
        scores = torch.einsum('bjnd,dd,bind->bjin', parent_hidden, self.b_1, child_hidden)
        # arc_num_scores = torch.einsum('bind, dd, bjnd->bijd', F.pad(parent_hidden, (0, 1), value=1), self.b_2, F.pad(child_hidden, (0, 1), value=1))
        # scores = torch.einsum('bjnd,dd,bind->bjin', F.pad(parent_hidden, (0, 1), value=1), self.b_1, F.pad(child_hidden, (0, 1), value=1))
        
        if self.n_head != 1:
            scores = self.onet(scores)
        scores = scores.squeeze(-1)
        scores = self.softmax(scores)
        arc_num_scores = (arc_num_scores[:, -1, :-1].sum(dim=-2) + arc_num_scores[:, :, -1].sum(dim=-2)) / np.sqrt(2 * arc_num_scores.shape[-2])
        arc_num_logits = self.prob(arc_num_scores)
        arc_num_prob = self.arc_softmax(arc_num_logits)
        return scores, arc_num_prob, arc_num_logits

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
        self.embed_k_net = nn.ModuleList([torch.nn.Linear(256, self.d_model) for _ in range(mixing_num)])

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
            # content_rel_embed = self.content_rel_embed
            if content_rel_embed is not None:
                r2 = torch.arange(max_len - min_len, -1, -1.0, device=w.device).long()
                W_k = self.qkv_net[0].weight[self.d_model:2 * self.d_model, :]
                # R_k = self.r_net.weight
                biases = [rel_embed(r2) for rel_embed in content_rel_embed]
                biases = [self.embed_k_net[i](biases[i]) for i in range(mixing_num)]
                # biases = [self.embed_k_net[i](biases[i]).repeat(1, self.n_head) for i in range(mixing_num)]
                rel_wk = [(bias@W_k.T).view(max_len - min_len + 1, self.n_head, self.d_head) for bias in biases]

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            # _, w_head_k, _ = torch.chunk(w_graphlayer, 3, dim=-1)
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
            # min_len = -75
            attn_relpos = torch.clip(attn_relpos, min_len, max_len).long()
            attn_relpos = attn_relpos.permute(0, 2, 3, 1)
            attn_relpos = (max_len - attn_relpos).long()
            Moderate_AC_rel = [torch.einsum('ibnd,jnd->ijbn', (rw_head_q, item)) for item in rel_wk]   # exchange position
            BD = self._rel_shift(BD)
            for idx, item in enumerate(Moderate_AC_rel):
                BD = BD + item.gather(1, attn_relpos[idx].unsqueeze(-1).expand(-1, -1, -1, BD.shape[-1]))

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

        # file_name = "./outputs/record_GiLT_small_0921v1_attn_prob_Writing"
        # tgt_len, src_len = np.load(f'{file_name}.mmap.shape.npy')
        # mmap = np.memmap(f'{file_name}.mmap', dtype=np.float32, mode='r+',
        #                 shape=(16, tgt_len, src_len))
        # # record_attn_prob = attn_prob[:, :, :, 7].detach().cpu().numpy()
        # record_attn_prob = torch.mean(attn_prob, dim=-1).detach().cpu().numpy()
        # if attn_prob.shape[0] == attn_prob.shape[1] and attn_prob.shape[0] != 1:
        #     mmap[0, :] = record_attn_prob[:tgt_len, :src_len, 0]
        #     mmap.flush()

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
                 rel_type = None):
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
            self.rel_embed = nn.ModuleList([torch.nn.Embedding(151, 256) for _ in range(mixing_num)])
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
            if rel_type == "degree" or rel_type == "mixing": # ablation
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
                                    degree_for_pointer[j] += 10 # 10
                                    degree_for_pointer[add_arc_idx] += 1

                            if right_sent_arrow[add_arc_idx - 1]: #i -> j
                                for j in right_sent_arrow[add_arc_idx - 1]:
                                    if j != 0:
                                        sent_attn_relpos_step[id_to_index[j]] += 1
                                    sent_attn_relpos_step[id_to_index[add_arc_idx]] += 10 # 10
                                    degree_for_pointer[j] += 1
                                    degree_for_pointer[add_arc_idx] += 10 # 10
                        sent_attn_relpos[i] = sent_attn_relpos_step[:]
                        sent_attn_relpos_for_pointer[add_arc_idx] = degree_for_pointer[:]
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
                    # Incremental_depth = IncrementalBellmanFord(graph_len)
                    # Incremental_depth.set_source(0)
                    sent_attn_relpos = np.zeros((length_i, length_i))
                    sent_attn_relpos_for_pointer = np.zeros((max(sent_index_to_id) + 1, (max(sent_index_to_id)) + 1))
                    sent_attn_relpos_for_pointer[:, 0] = 1  # root
                    depth_res = []
                    for i in range(len(left_sent_arrow)):
                        if left_sent_arrow[i]:  #i <- j
                            for j in left_sent_arrow[i]:
                                # Incremental_depth.add_edge(j, i + 1, 1)
                                # Incremental_depth.add_edge(i + 1, j, 1)
                                graph[j][i + 1] = 1
                                graph[i + 1][j] = 1
                        if right_sent_arrow[i]: #i -> j
                            for j in right_sent_arrow[i]:
                                # Incremental_depth.add_edge(i + 1, j, 1)
                                # Incremental_depth.add_edge(j, i + 1, 1)
                                graph[i + 1][j] = 1
                                graph[j][i + 1] = 1
                        # cur_depth = Incremental_depth.get_depth_for_GiLT()
                        # depth_res.append(cur_depth)
                    
                    last_finished_word_idx = None
                    finished_word_idx = None
                    for i in range(input_size):
                        if sent_index_to_id[i] == -1:
                            if finetune is not None:
                                last_finished_word_idx = finished_word_idx
                            continue
                        finished_word_idx = sent_index_to_id[i] # first token max(sent_index_to_id[i - 1], 0)
                        newgraph = graph[:finished_word_idx + 1, :finished_word_idx + 1].tolist()
                        depth = calculate_depth(newgraph)
                        # depth = depth_res[finished_word_idx - 1]
                        if last_finished_word_idx:
                            depth[1:last_finished_word_idx + 1] = [0] * last_finished_word_idx
                        # if rel_type == "rel_depth":
                        # depth = [item - depth[finished_word_idx] for item in depth]
                        for id, depth_value in enumerate(depth[1:]):
                            sent_attn_relpos[i, id_to_index[id + 1]] = depth_value
                        # sent_attn_relpos_for_pointer[finished_word_idx] = np.array(depth)
                        sent_attn_relpos_for_pointer[finished_word_idx, :len(depth)] = depth[:]
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
                    # Incremental_distance = IncrementalBellmanFord(graph_len)
                    sent_attn_relpos = np.zeros((length_i, length_i))
                    sent_attn_relpos_for_pointer = np.zeros((max(sent_index_to_id) + 1, (max(sent_index_to_id)) + 1))
                    distance_res = []
                    for i in range(len(left_sent_arrow)):
                        # Incremental_distance.set_source(i + 1)
                        if left_sent_arrow[i]:  #i <- j
                            for j in left_sent_arrow[i]:
                                # Incremental_distance.add_edge(j, i + 1, 10)
                                # Incremental_distance.add_edge(i + 1, j, 1)
                                graph[j][i + 1] = 10 # 10
                                graph[i + 1][j] = 1
                        if right_sent_arrow[i]: #i -> j
                            for j in right_sent_arrow[i]:
                                # Incremental_distance.add_edge(i + 1, j, 10)
                                # Incremental_distance.add_edge(j, i + 1, 1)
                                graph[i + 1][j] = 10 # 10
                                graph[j][i + 1] = 1
                        # cur_distance= Incremental_distance.get_dist_for_GiLT()
                        # distance_res.append(cur_distance)

                    last_finished_word_idx = None
                    finished_word_idx = None
                    for i in range(input_size):
                        if sent_index_to_id[i] == -1:
                            if finetune is not None:
                                last_finished_word_idx = finished_word_idx
                            continue
                        finished_word_idx = sent_index_to_id[i] # first token max(sent_index_to_id[i - 1], 0)
                        newgraph = graph[:finished_word_idx + 1, :finished_word_idx + 1].tolist()
                        distance = dijkstra_heap(newgraph, finished_word_idx)
                        # distance = distance_res[finished_word_idx - 1]
                        if last_finished_word_idx:
                            distance[1:last_finished_word_idx + 1] = [0] * last_finished_word_idx
                        for id, distance_value in enumerate(distance[1:]):
                            sent_attn_relpos[i, id_to_index[id + 1]] = distance_value
                        # sent_attn_relpos_for_pointer[finished_word_idx] = np.array(distance)
                        sent_attn_relpos_for_pointer[finished_word_idx, :len(distance)] = distance[:]
                    # for j in range(length_i - len(sent_attn_relpos)):
                        # sent_attn_relpos.append([0]*(length_i))
                    attn_relpos_for_pointer.append(sent_attn_relpos_for_pointer[:-1])
                    attn_relpos.append(sent_attn_relpos)
            if rel_type == "predicate_depth": # or rel_type == "mixing"
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
                attn_relpos = attn_relpos.reshape(mixing_num, batch, length_i, length_i)
                attn_relpos_for_pointer = [torch.LongTensor(np.array([attn_relpos_for_pointer[i + j * batch] for j in range(mixing_num)])) for i in range(batch)]
            if torch.cuda.is_available():
                attn_relpos = attn_relpos.cuda()
                attn_relpos_for_pointer = [attn_relpos_for_pointer[i].cuda() for i in range(batch)]

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
            if finetune is not None and i <= 11:
                core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, attn_mask=attn_mask,
                    attn_relpos=None, min_len=min_relative_length, max_len=max_relative_length,
                    rel_embed=None)
            else:
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
        if finetune is not None and finetune != "sts":
            mask = torch.zeros_like(loss)
            for j, sent in enumerate(x):
                if finetune == "sst2":
                    idx = find_label_idx(sent, [26858, 5599]) + 1
                elif finetune == "mrpc" or finetune == "rte":
                    idx = find_label_idx(sent, [2745, 11346, 5599]) + 2
                mask[idx, j] = 1
            loss = loss * mask

        loss = loss.permute(1, 0).contiguous()
        loss = loss.sum(1)

        if return_h:
            if finetune == "sts":
                STS_output = self.STS(hiddens[-1][[len(sent) - 2 for sent in x], [i for i in range(batch)]])
                return STS_output, torch.cat([hiddens[9], hiddens[15]], dim = -1), word_emb, attn_relpos_for_pointer, prob
            if use_mask == "graphlayer":
                return loss, torch.cat([hiddens[self.num_layers // 2 + 1], hiddens[self.num_layers - 1]], dim = -1), word_emb, attn_relpos_for_pointer, prob    #torch.cat([hiddens[9], hiddens[15]], dim = -1)
                # return loss, (hiddens[9] + hiddens[15]) / 2
            return loss, core_out, prob
        elif finetune == "sts":
            STS_output = self.STS(hiddens[-1][[len(sent) - 2 for sent in x], [i for i in range(batch)]])
            return STS_output, prob
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

            return prob, new_keys, new_values, torch.cat([hiddens[9], hiddens[15]], dim = -1)  # (hiddens[9] + hiddens[15]) / 2
    
    def TXL_inference(self, x, past_keys, past_values, seq_len):
        with torch.no_grad():
            inp = x.permute(1, 0).contiguous()  # 1 * batch new token
            word_emb = self.emb(inp)
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
                                    attn_mask=None, attn_relpos=None, 
                                    min_len=0, max_len=150, rel_embed = None,
                                    past_keys=past_keys_p[i], past_values=past_values_p[i], cache=True)
                else:
                    core_out, new_key, new_value = \
                                    layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, 
                                    attn_mask=None, attn_relpos=None, 
                                    min_len=0, max_len=150, rel_embed = None,
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

            return prob, new_keys, new_values
    

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
