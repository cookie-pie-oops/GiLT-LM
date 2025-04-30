import math
import torch

def tensor_memory(tensor):
    mem_bytes = tensor.detach().numel() * tensor.detach().element_size()
    mem_mb = mem_bytes / (1024 ** 2)
    return mem_bytes, mem_mb


"""
For Syntactic Generalization Test
"""
def eval_math_expr(expr):
    try:
        return eval(expr)
    except:
        return math.nan

def get_suite_type(name):
    if "npi" in name or "reflexive" in name:
        suite_type = "licensing"
    elif "mvrr" in name or "npz" in name:
        suite_type = "garden-path effects"
    elif "center_embed" in name:
        suite_type = "center embedding"
    elif "subordination" in name:
        suite_type = "gross syntactic expectation"
    elif "fgd" in name or "cleft" in name:
        suite_type = "long-distance dependencies"
    elif "number" in name:
        suite_type = "agreement"
    else:
        suite_type = "other"
    return suite_type