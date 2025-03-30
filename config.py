from dataclasses import dataclass

@dataclass
class ModelConfig:
    w_dim: int = 1024
    n_head: int = 8
    d_head: int = 128
    d_inner: int = 4096
    num_layers: int = 16
    dropout: float = 0 # txl 0.1?
    dropoutatt: float = 0 # txl 0.1?
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    stack_pad_id: int = -100
    pre_lnorm: bool = False
    max_stack_depth: int = 200
    
@dataclass
class ParallelConfig:
    parallel: str = "dp" # "ddp" or "dp" or "none"
    local_rank: int = 0
    assert parallel in ["ddp", "dp", "none"], f"parallel must be one of ['ddp', 'dp', 'none'], but got {parallel}"
    if parallel == "ddp":
        raise NotImplementedError("ddp is not implemented yet")
    # elif parallel == "dp":
    #     # we have dim alignment bugs in dp
    #     raise NotImplementedError("dp is not implemented yet")

@dataclass
class TrainConfig:
    run_name = "push_bllip_con_test"
    
    seed: int = 12345
    batch_size: int = 64
    num_workers: int = 8
    epochs: int = 4 # debug: 110, normal: 4
    
    max_lr: float = 3e-4 # debug: 1e-2
    start_lr: float = 1e-7
    warmup_steps: int = 8000 # debug: 100, normal: 8000
    eta_min: float = 3e-7
    
    weight_decay: float = 0 # debug: 0.01, normal: 0
    
    # while training, the ratio of attachment loss to word loss.
    # 1/(1+ratio)*loss_word+ratio/(1+ratio)*loss_attachment is the training loss
    # while finding eval ppl, we only use exp(word_loss+attachment_loss) & exp(word_loss), not the ratio
    attachment_ratio: float = 1.0
    
    max_grad_norm: float = 3.0
    gradient_accumulation_steps: int = 1 # 
    log_interval: int = 1 # debug: 1, normal: 100
    eval_interval: int = 1 # debug: 1, normal: 1000
    
    
    # DEBUG
    run_name = "push_bllip_con_debug"
    epochs: int = 110
    start_lr: float = 1e-6
    max_lr: float = 1e-4
    warmup_steps: int = 100
    log_interval: int = 1
    eval_interval: int = 1