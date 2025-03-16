import torch
import torch.nn.functional as F

def _shift(posmat: torch.Tensor) -> torch.Tensor:
    p = F.pad(posmat, (0, 1, 0, 1)).flatten(-2)
    p = p.narrow(-1, posmat.shape[-1] // 2, posmat.shape[-1] * posmat.shape[-2]).view_as(posmat)
    return p.narrow(-1, 0, (posmat.shape[-1] + 1) // 2)

# 构造一个模拟张量 [n_batch=1, n_head=2, n_out=3, n_in * 2 - 1=5] 
x = torch.arange(1, 1+1*2*3*5, dtype=torch.float).view(1, 2, 3, 5)
print("原始 x:", x, x.shape, "\n")

result = _shift(x)
print("移位后:", result, result.shape)