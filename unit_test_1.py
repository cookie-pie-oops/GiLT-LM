import unittest
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from model_bllip_con import PushdownTransformerConstituency

class TestPushdownTransformerConstituency(unittest.TestCase):

    def setUp(self):
        # 使用较小的参数进行测试，便于快速运行
        self.vocab_size = 32000
        self.w_dim = 1024
        self.n_head = 8
        self.d_head = 128
        self.d_inner = 4096
        self.num_layers = 16  # 测试时使用较少层数
        self.dropout = 0.1
        self.dropoutatt = 0.1
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.max_stack_depth = 150  # 测试时使用较小的堆栈深度上限

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PushdownTransformerConstituency(
            vocab_size=self.vocab_size,
            w_dim=self.w_dim,
            n_head=self.n_head,
            d_head=self.d_head,
            d_inner=self.d_inner,
            num_layers=self.num_layers,
            dropout=self.dropout,
            dropoutatt=self.dropoutatt,
            pad_id=self.pad_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            stack_pad_id=-100,
            pre_lnorm=False,
            max_stack_depth=self.max_stack_depth
        ).to(self.device)
        self.model.apply(self.weights_init)  # 初始化权重
        
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight'):
                scale = 2.0 / self.num_layers
                fan_in = nn.init._calculate_correct_fan(m.weight, 'fan_in')
                nn.init.trunc_normal_(m.weight, 0.0, np.sqrt(scale / fan_in))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('PushdownTransformerConstituency') != -1:
            if hasattr(m, 'r_w_bias'):
                fan_in = nn.init._calculate_correct_fan(m.r_w_bias, 'fan_in')
                nn.init.trunc_normal_(m.r_w_bias, 0.0, np.sqrt(1.0 / fan_in))
            if hasattr(m, 'r_r_bias'):
                fan_in = nn.init._calculate_correct_fan(m.r_r_bias, 'fan_in')
                nn.init.trunc_normal_(m.r_r_bias, 0.0, np.sqrt(1.0 / fan_in))
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                fan_in = nn.init._calculate_correct_fan(m.weight, 'fan_in')
                nn.init.uniform_(m.weight, -np.sqrt(3.0 / fan_in), np.sqrt(3.0 / fan_in))

    def generate_dummy_inputs(self, T, B):
        """
        生成测试所需的伪造数据：
          - data: [T, B] 的 token id 序列（输入序列，长度 T）
          - target: [T, B] 的 token id 序列（预测目标，对应 seq[1:T+1]）
          - stack_tape: [B, T, T] 的堆栈深度信息（每个元素均在 [0, max_stack_depth) 内）
          - attachment_labels: [B, T] 的 reduce-with attachment 标签，对于 time [:, t], 值 <= t+1
          - attachment_mask: [B, T, T+1] 的二值 mask（1 表示有效）
        """
        # data = torch.randint(low=0, high=self.vocab_size, size=(T, B))
        # target = torch.randint(low=0, high=self.vocab_size, size=(T, B))
        x = torch.randint(low=0, high=self.vocab_size, size=(T+1, B))
        data = x[:-1]
        target = x[1:]
        stack_tape = torch.randint(low=0, high=self.max_stack_depth, size=(B, T, T))
        attachment_labels = torch.zeros(B, T, dtype=torch.long)
        for t in range(T):
            attachment_labels[:, t] = torch.randint(low=0, high=t+1, size=(B,))
        # print(T+1 in attachment_labels)
        # attachment_mask = torch.ones(B, T, T+1, dtype=torch.uint8)
        # causal. for row t, (from 0 to T-1), <= t+1. i.e. column of 1 | tril
        # attachment_mask = torch.tril(torch.ones(T, T+1), diagonal=1).unsqueeze(0).expand(B, T, T+1) 
        # be like
        # 1 1 0 0 0
        # 1 1 1 0 0
        # 1 1 1 1 0
        # 1 1 1 1 1
        
        # to device
        data = data.to(self.device)
        target = target.to(self.device)
        stack_tape = stack_tape.to(self.device)
        attachment_labels = attachment_labels.to(self.device)
        # attachment_mask = attachment_mask.to(self.device)
        
        # print every shape
        print(f"batch size: {B}")
        print(f"length: {T}")
        print(f"data shape: {data.shape}")
        print(f"target shape: {target.shape}")
        print(f"stack_tape shape: {stack_tape.shape}")
        print(f"attachment_labels shape: {attachment_labels.shape}")
        # print(f"attachment_mask shape: {attachment_mask.shape}") 
        
        return data, target, stack_tape, attachment_labels

    def test_forward_loss(self):
        """测试前向传播返回的损失为标量，并且不会报错。"""
        T, B = 7, 3
        data, target, stack_tape, attachment_labels = self.generate_dummy_inputs(T, B)
        loss1, loss2 = self.model(data, target, stack_tape, attachment_labels)
        loss = loss1 + loss2
        print("LOSS: ", loss)
        # loss should be scalar
        self.assertTrue(loss.dim() == 0, "The loss must be scalar")

    def test_forward_return_hidden(self):
        """测试 forward(return_h=True) 能返回 (loss, hidden)，并验证 hidden 的形状。"""
        T, B = 5, 2
        data, target, stack_tape, attachment_labels = self.generate_dummy_inputs(T, B)
        loss1, loss2, hidden = self.model(data, target, stack_tape, attachment_labels, return_h=True)
        loss = loss1 + loss2
        self.assertTrue(loss.dim() == 0, "The loss must be scalar")
        # hidden should be [T, B, w_dim]
        self.assertEqual(hidden.shape, (T, B, self.w_dim), "Hidden Shape Mismatch")

    def test_backward(self):
        """测试反向传播是否能正常进行（检查梯度是否能正常计算）。"""
        T, B = 6, 2
        data, target, stack_tape, attachment_labels = self.generate_dummy_inputs(T, B)
        loss1, loss2 = self.model.forward(data, target, stack_tape, attachment_labels)
        # Test backward
        loss = loss1 + loss2
        self.model.zero_grad()
        loss.backward()
        # 随便检查模型中一个参数是否有梯度
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Param {name} has no gradient")
                break

    def test_invalid_stack_tape_shape(self):
        """测试传入错误形状的 stack_tape 时，模型是否能报错（或计算结果不符合预期）。
           这里给出一个形状为 [B, T+1, T] 的 stack_tape，与预期 [B, T, T] 不符。
        """
        T, B = 5, 2
        data, target, _, attachment_labels = self.generate_dummy_inputs(T, B)
        # 故意构造一个错误的 stack_tape
        stack_tape_invalid = torch.randint(low=0, high=self.max_stack_depth, size=(B, T+1, T))
        with self.assertRaises(Exception):
            # 这里可能会由于维度不匹配等原因报错，捕捉异常即可
            _ = self.model(data, target, stack_tape_invalid, attachment_labels)

if __name__ == '__main__':
    unittest.main()
