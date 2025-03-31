def tensor_memory(tensor):
    mem_bytes = tensor.numel() * tensor.element_size()
    mem_mb = mem_bytes / (1024 ** 2)
    return mem_bytes, mem_mb