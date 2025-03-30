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
    parallel: str = "1" # "ddp" or "dp" or "none"
    local_rank: int = 0
    assert parallel in ["ddp", "dp", "none"], f"parallel must be one of ['ddp', 'dp', 'none'], but got {parallel}"

@dataclass
class TrainConfig:
    run_name = "push_bllip_con_test"
    
    seed: int = 42
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    num_workers: int = 8
    
    max_grad_norm: float = 1.5
    gradient_accumulation_steps: int = 1
    log_interval: int = 100
    # save_interval: int = 1000
    eval_interval: int = 1000