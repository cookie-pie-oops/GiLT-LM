from model_bllip_con import PushdownTransformerConstituency
from config import ModelConfig
import sentencepiece as spm
import torch

PATH = "ckpt/push_bllip_con_test_gas4/best/model.pt"
sp = spm.SentencePieceProcessor()
sp.Load("./data_process/spm_parsing/BLLIP_spm.model")
model_args = ModelConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the model
model = PushdownTransformerConstituency(
    vocab_size=sp.GetPieceSize(),
    w_dim=model_args.w_dim,
    n_head=model_args.n_head,
    d_head=model_args.d_head,
    d_inner=model_args.d_inner,
    num_layers=model_args.num_layers,
    dropout=model_args.dropout,
    dropoutatt=model_args.dropoutatt,
    pad_id=model_args.pad_id,
    bos_id=model_args.bos_id,
    eos_id=model_args.eos_id,
    stack_pad_id=model_args.stack_pad_id,
    pre_lnorm=model_args.pre_lnorm,
    max_stack_depth=model_args.max_stack_depth
).to(device)
# Load the state dict
state_dict = torch.load(PATH, map_location=device, weights_only=True)
# Load the model state dict
# print(state_dict.keys())
# exit()

# keys: "module.xxx" -> "xxx"
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

# random gen
ids = sp.EncodeAsIds("The quick brown fox jumps over the lazy dog.")
ids = torch.tensor(ids, device=device).unsqueeze(0) # (1, 9)
ids = ids.repeat(2, 1)
step = 5
# stack tape shape [1, 5, 5] randint
stack_tape = torch.randint(0, model_args.max_stack_depth, (2, step, step), device=device)
# say, 1 set of {1, 3} -> list_reduced
list_reduced = [
    set([1, 3]), # 1st beam
    set([1, 2]), # 2nd beam
]
a, b = model.take_step_silver_tree(
    ids=ids,
    stack_tape=stack_tape,
    list_reduced=list_reduced,
    step=step,
)
print(a,b)