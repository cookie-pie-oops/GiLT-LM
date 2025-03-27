import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PushdownAttention(nn.Module):
    """
    PushdownAttention 实现了基于堆栈深度信息修正 key 的自注意力层，
    参考自论文附录 Figure 7。

    输入：
      - x: (B, T, n_embd) 的输入隐藏状态序列
      - stack_tape: (B, T) 的堆栈深度（整数张量），每个位置存储该 token 的深度
    输出：
      - 返回经过堆栈信息加权后的输出 (B, T, n_embd)
    """

    def __init__(self, n_embd, n_head, attn_dropout=0.1, resid_dropout=0.1, max_depth=50, max_seq_len=1024):
        super(PushdownAttention, self).__init__()
        self.n_embd = n_embd  # state_size
        self.n_head = n_head  # num_heads
        self.head_dim = n_embd // n_head  # state_size // num_heads

        # 一次性线性映射到 q, k, v (每个大小 n_embd)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # 输出投影
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

        # 用于将堆栈深度映射为向量，维度和 head_dim 一致
        self.beta = nn.Embedding(max_depth, self.head_dim)

        # 预生成因果 mask，此处假定最大序列长度为 max_seq_len
        # bias 的形状为 (1, 1, max_seq_len, max_seq_len)
        self.register_buffer("bias", torch.tril(
            torch.ones(1, 1, max_seq_len, max_seq_len)))

    def forward(self, x, stack_tape):
        B, T, C = x.size()
        # 计算 q, k, v，形状均为 (B, T, n_embd)
        qkv = self.c_attn(x)
        q, k, v = torch.split(qkv, C, dim=2)
        # 重塑为多头形式 (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 通过堆栈 tape 得到深度嵌入：
        # stack_tape: (B, T, T) -> beta 映射后为 (B, T, T, head_dim)
        # unsqueeze 后变为 (B, 1, T, T, head_dim)
        depth_emb = self.beta(stack_tape.long()).unsqueeze(1)

        # 将 key 增强：对 k 先在 dim=2 扩展，再加上深度嵌入（同样在 dim=2 扩展）
        # k.unsqueeze(2): (B, n_head, 1, T, head_dim)
        # depth_emb: (B, 1, T, T, head_dim)
        augmented_keys = k.unsqueeze(2) + depth_emb
        # 缩放
        augmented_keys = augmented_keys / math.sqrt(self.head_dim)

        # 计算注意力得分：
        # 将 q 扩展到 (B, n_head, T, 1, head_dim)
        # 将 augmented_keys 转置最后两个维度 (B, n_head, T, head_dim, T)
        # 相乘后得到 (B, n_head, T, 1, T)，再去掉 1 得到 (B, n_head, T, T)
        att_logits = (q.unsqueeze(
            3) @ augmented_keys.transpose(-2, -1)).squeeze(3)

        # 使用因果 mask（bias）屏蔽未来信息
        bias = self.bias[:, :, :T, :T]  # (1, 1, T, T)
        att_logits = att_logits.masked_fill(bias == 0, float("-inf"))

        # softmax、dropout，并加权求和得到输出
        att = F.softmax(att_logits, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, n_head, T, head_dim)

        # 将所有头拼接回 (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class AttachmentHead(nn.Module):
    """
    AttachmentHead 实现了根据当前隐藏状态和下一个预测 token，
    利用注意力机制选择一个候选 constituent（通过右侧 token）进行 reduce，
    参考自论文附录 Figure 8。

    输入：
      - x: (B, T, n_embd) 隐藏状态序列
      - stack_tape: (B, T) 的堆栈深度（整数张量）
      - next_word: (B, T, embd_dim) 表示当前新预测的 token（经过 MLP 得到的向量）
    输出：
      - 返回形状为 (B, T, T+1) 的 attachment logits（经过因果 mask 后）
    """

    def __init__(self, n_embd, embd_dim, max_depth=50):
        super(AttachmentHead, self).__init__()
        self.n_embd = n_embd
        self.embd_dim = embd_dim
        # 将输入 x 映射到 query 和 key（拼接后维度为 2*embd_dim）
        self.data_to_qk = nn.Linear(n_embd, 2 * embd_dim)
        # 用于计算 next_word 的 query 与 key 的 MLP，
        # 输入为拼接后的 [q, next_word]，输出维度为 embd_dim
        self.q_next_word_mlp = nn.Linear(2 * embd_dim, embd_dim)
        self.k_next_word_mlp = nn.Linear(2 * embd_dim, embd_dim)
        # 将 key 与堆栈深度信息拼接后映射回 embd_dim
        self.key_and_stack_mlp = nn.Linear(2 * embd_dim, embd_dim)
        # 堆栈深度嵌入
        self.beta = nn.Embedding(max_depth, embd_dim)
        # 这里 bias 我们在 forward 中动态生成因果 mask

    def forward(self, x, stack_tape, next_word):
        # next_word 是当前预测的 token，形状为 (B, T, embd_dim)
        # x 是隐藏状态序列，形状为 (B, T, n_embd)
        # stack_tape: (B, T, T) 的堆栈深度信息

        B, T, C = x.size()
        # 得到 q 和 k，形状均为 (B, T, embd_dim)
        qk = self.data_to_qk(x)
        q, k = torch.split(qk, self.embd_dim, dim=2)

        # 计算 next_word 的 query 与 key
        # 拼接 [q, next_word]，假设 next_word 的最后一维与 embd_dim 相同
        cat_inp = torch.cat([q, next_word], dim=-1)  # (B, T, 2*embd_dim)
        next_word_q = self.q_next_word_mlp(cat_inp)    # (B, T, embd_dim)
        next_word_k = self.k_next_word_mlp(cat_inp)      # (B, T, embd_dim)

        # 将 k 扩展到每个目标位置： (B, T(klen), embd_dim) -> (B, 1, T(klen), embd_dim) -> (B, T, T, embd_dim)
        k_exp = k.unsqueeze(1).repeat(1, T, 1, 1)

        # 计算堆栈深度嵌入：stack_tape (B, T, T) -> (B, T, T, embd_dim)
        depth_emb = self.beta(stack_tape.long()).unsqueeze(
            1)

        # 拼接 k 与深度信息，通过 MLP 得到融合后的 k 信息
        k_with_info = self.key_and_stack_mlp(
            torch.cat([k_exp, depth_emb], dim=-1))
        k_with_info = k_with_info / math.sqrt(self.embd_dim)

        # 计算 attachment logits
        # next_word_q: (B, T, embd_dim) -> unsqueeze为 (B, T, 1, embd_dim)
        # k_with_info: (B, T, T, embd_dim) -> 转置最后两维得到 (B, T, embd_dim, T)
        attach_logits = (next_word_q.unsqueeze(
            2) @ k_with_info.transpose(-2, -1)).squeeze(2)  # (B, T, T)

        # 计算 self-attachment 得分（no-reduce 情况）
        next_word_k = next_word_k / math.sqrt(self.embd_dim)
        logits_self = (next_word_q.unsqueeze(
            2) @ next_word_k.unsqueeze(3)).squeeze(2)  # (B, T, 1)

        # 在 attach_logits 最后拼接一列 0 值，使其形状变为 (B, T, T+1)
        pad_tensor = torch.zeros(B, T, 1, device=attach_logits.device)
        attach_logits_l = torch.cat(
            [attach_logits, pad_tensor], dim=-1)  # (B, T, T+1)

        # 构造 scatter 的 indices，形状为 (B, T, 1)，每个位置填入 1,2,...,T
        indices = (1 + torch.arange(T, device=attach_logits.device)
                   ).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)
        # 将 logits_self 插入到第 k+1 个位置
        logits = attach_logits_l.scatter(
            2, indices, logits_self)  # (B, T, T+1)

        # 在 logits 的第一行前填充一行全 0，使其形状变为 (B, T+1, T+1)
        zeros_row = torch.zeros(B, 1, logits.size(2), device=logits.device)
        logits = torch.cat([zeros_row, logits], dim=1)  # (B, T+1, T+1)

        # 构造因果 mask：下三角 mask，形状为 (1, T+1, T+1)
        mask = torch.tril(torch.ones(
            T+1, T+1, device=logits.device)).unsqueeze(0)
        logits = logits.masked_fill(mask == 0, float("-inf"))

        # 去掉最上面一行，返回 (B, T, T+1)
        return logits[:, 1:]


# ========== 1) 模拟一些超参数 ==========
B = 2          # batch size
T = 5          # 序列长度
n_embd = 16    # PushdownAttention 的输入/输出维度
embd_dim = 16  # AttachmentHead 内部使用的维度
n_head = 2     # multi-head 个数
max_depth = 10
max_seq_len = 10  # 用于mask大小

# ========== 2) 实例化上述模块 ==========

pushdown_attn = PushdownAttention(
    n_embd=n_embd,
    n_head=n_head,
    max_depth=max_depth,
    max_seq_len=max_seq_len
)
attachment_head = AttachmentHead(
    n_embd=n_embd,
    embd_dim=embd_dim,
    max_depth=max_depth
)

# ========== 3) 准备输入张量 ==========

# 随机初始化输入序列 x，形状 (B, T, n_embd)
x = torch.randn(B, T, n_embd)

# 假设初始堆栈深度全为 0，形状 (B, T)
stack_tape = torch.zeros(B, T, dtype=torch.long)

# 下一个“新预测词”的向量表示 (为了演示，这里随意用一个 MLP 从 x 的最后状态生成)
mlp_for_next_token = nn.Linear(n_embd, embd_dim)
# 先取序列最后一个位置作为代表
x_last = x[:, -1, :]                   # (B, n_embd)
next_word_embed_single = mlp_for_next_token(x_last)  # (B, embd_dim)

# 为了演示 AttachmentHead 的输入，扩展到 (B, T, embd_dim)
# 在真实场景中，你可能只需要 (B, 1, embd_dim) 表示“单个”新词
# 这里为了能跟 AttachmentHead 的 (B, T, embd_dim) 对上，做一次扩展
next_word_embed = next_word_embed_single.unsqueeze(1).expand(B, T, embd_dim)

# ========== 4) 调用 PushdownAttention ==========

attn_out = pushdown_attn(x, stack_tape)
# 结果形状: (B, T, n_embd)

# ========== 5) 调用 AttachmentHead ==========

attach_logits = attachment_head(attn_out, stack_tape, next_word_embed)
# 结果形状: (B, T, T+1)

# ========== 6) 得到 attachment 分布 (可选) ==========

attach_probs = F.softmax(attach_logits, dim=-1)

# 打印结果形状进行检查
print("attn_out.shape:     ", attn_out.shape)     # (2, 5, 16)
print("attach_logits.shape:", attach_logits.shape)  # (2, 5, 6)
print("attach_probs.shape: ", attach_probs.shape)  # 同上
