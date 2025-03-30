from dataclasses import dataclass

@dataclass
class ModelConfig:
    w_dim: int = 380
    n_head: int = 10
    d_head: int = 38
    d_inner: int = 900
    num_layers: int = 16
    dropout: float = 0.1
    dropoutatt: float = 0.1
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    stack_pad_id: int = -100
    pre_lnorm: bool = False
    max_stack_depth: int = 200
    
@dataclass
class ParallelConfig:
    parallel: str = "none" # "ddp" or "dp" or "none"
    local_rank: int = 0
    assert parallel in ["ddp", "dp", "none"], f"parallel must be one of ['ddp', 'dp', 'none'], but got {parallel}"
    if parallel == "ddp":
        raise NotImplementedError("ddp is not implemented yet")

@dataclass
class TrainConfig:
    run_name = "push_bllip_con_debug"
    
    seed: int = 42
    batch_size: int = 64
    epochs: int = 111 # debug: 111, normal: 10
    lr: float = 3e-4 # debug: 1e-2
    warmup_steps: int = 100 # debug: 1, normal: 1000
    weight_decay: float = 0.01
    num_workers: int = 8
    
    max_grad_norm: float = 1.5
    gradient_accumulation_steps: int = 1
    log_interval: int = 1 # debug: 1, normal: 100
    eval_interval: int = 1 # debug: 1, normal: 1000