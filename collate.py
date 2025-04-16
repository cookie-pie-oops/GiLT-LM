from config import ModelConfig
import torch
model_args = ModelConfig()
def collate_fn(batch):
    # find max len
    max_len = max(len(item["ids"]) for item in batch)
    
    padded_ids = []
    padded_stack_tape = []
    padded_attachment_labels = []
    lengths = []
    idxs = []
    
    pad_id = model_args.pad_id  # pad_id: 0 for spm vocab
    pad_value_stack = model_args.max_stack_depth - 1  # stack_tape 的 pad 值: max_depth - 1 (-100 will cause index error, and ?0 has been occupied?)
    pad_value_att = model_args.stack_pad_id    # attachment_labels 的 pad 值: -100 (0 has been occupied)
    
    # for item in batch:
    #     T = len(item["ids"])
    #     lengths.append(T)
    #     idxs.append(item.get("idx", 0))
        
    #     # 对 ids 进行 padding 到 max_len
    #     padded_ids.append(item["ids"] + [pad_id] * (max_len - T))
        
    #     # 对 stack_tape 进行 padding：原始是 T x T，需要 pad 到 max_len x max_len
    #     # 先初始化一个 max_len x max_len 的矩阵，用 pad_value_stack 填充
    #     padded_matrix = [[pad_value_stack] * max_len for _ in range(max_len)]
    #     for i in range(T):
    #         for j in range(T):
    #             padded_matrix[i][j] = item["stack_tape"][i][j]
    #     padded_stack_tape.append(padded_matrix)
        
    #     # 对 attachment_labels 进行 padding 到 max_len
    #     padded_attachment_labels.append(item["attachment_labels"] + [pad_value_att] * (max_len - T))
    
    for item in batch:
        T = len(item["ids"])
        lengths.append(T)
        idxs.append(item.get("idx", 0))
        
        # ids padding -> max_len
        padded_ids.append(item["ids"] + [pad_id] * (max_len - T))
        
        # pad stack_tape using tensor operations
        stack_tensor = torch.tensor(item["stack_tape"], dtype=torch.long)
        # full matrix with pad_value_stack
        padded_tensor = torch.full((max_len, max_len), pad_value_stack, dtype=torch.long)
        # fill the upper left corner with stack_tensor
        padded_tensor[:T, :T] = stack_tensor
        padded_stack_tape.append(padded_tensor) # so padded_stack_tape is a list of tensors
        
        # attachment_labels 的 padding
        padded_attachment_labels.append(item["attachment_labels"] + [pad_value_att] * (max_len - T))

    
    # to tensor
    
    batch_ids = torch.tensor(padded_ids, dtype=torch.long) # shape: [batch_size, max_len]
    
    batch_lengths = torch.tensor(lengths, dtype=torch.long) # shape: [batch_size]
    # batch_stack_tape = torch.tensor(padded_stack_tape, dtype=torch.long) # shape: [batch_size, max_len, max_len]
    batch_stack_tape = torch.stack(padded_stack_tape, dim=0) # shape: [batch_size, max_len, max_len]
    batch_attachment_labels = torch.tensor(padded_attachment_labels, dtype=torch.long) # shape: [batch_size, max_len]
    batch_idxs = torch.tensor(idxs, dtype=torch.long) # shape: [batch_size]
    del padded_ids, padded_stack_tape, padded_attachment_labels, lengths, idxs
    
    return {
        "ids": batch_ids,
        "lengths": batch_lengths,
        "idxs": batch_idxs,
        "stack_tape": batch_stack_tape,
        "attachment_labels": batch_attachment_labels
    }