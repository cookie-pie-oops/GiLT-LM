import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2Tokenizer, GPT2LMHeadModel #4.44.1
from typing import Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import os
import json
import csv
import copy
from torch.nn import CrossEntropyLoss
from train_graphLayer import load_data, load_multiarrow, predicate_alignment
from model_bllip_dep import calculate_depth, dijkstra, BiaffineAttention
from helping_utils.logger import configure_logger, get_logger
logger = get_logger()

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
mixing_num = 3
tokenizer = GPT2Tokenizer.from_pretrained("/home/huangty/GPT2/medium355M")
tokenizer.add_prefix_space = True
vocab = tokenizer.get_vocab()
startofword_id = [0 for _ in range(len(vocab))]
for k, v in vocab.items():
    if k.startswith("Ġ"):
        startofword_id[v] = 1

def synchronize_arrows(input_id):
    batch_sent_id = []
    bos_eos_id = 50256
    vocab_size = len(vocab)
    for sent in input_id:
        sent_id = []
        sent_startofword = [vocab_size if startofword_id[word] == 1 else word for word in sent]
        count_num = 0
        for idx, word_id in enumerate(sent_startofword):
            # if sent[idx] == 0:
            #     continue
            if word_id == bos_eos_id:
                sent_id.append(-1)
            elif word_id == vocab_size:
                count_num += 1
                sent_id.append(count_num)
            else:
                sent_id.append(count_num)
        batch_sent_id.append(sent_id)
    return batch_sent_id

class GiLTGPT2Attention(GPT2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_k_net = torch.nn.Linear(256 * mixing_num, 1024)
    
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        attn_relpos = None
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim) # (b, nhead, l, d)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if attn_relpos is not None: # relpos is embedding (b, l, l, 256 * mixing num)
            biases = self.embed_k_net(attn_relpos) # (b, l, l, 1024)
            new_shape = biases.shape[:-1] + (self.num_heads, self.head_dim)
            biases = biases.reshape(new_shape)
            biases = biases.permute(0, 3, 1, 2, 4)
            augmented_key = key.unsqueeze(2) + biases
            augmented_key = augmented_key / math.sqrt(self.head_dim)
            augmented_attn = (query.unsqueeze(3) @ augmented_key.transpose(-2,-1)).squeeze(3)

            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(augmented_attn.dtype).min
            mask_value = torch.full([], mask_value, dtype=augmented_attn.dtype, device=augmented_attn.device)
            attn_weights = torch.where(causal_mask, augmented_attn, mask_value)

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = attn_weights @ value
        else:
            raise NotImplementedError

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class GiLTGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=None)
        self.attn = GiLTGPT2Attention(config=config, layer_idx=layer_idx)
    
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        attn_relpos = None
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            attn_relpos=attn_relpos
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)

class GiLTGPT2(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.rel_embed = torch.nn.ModuleList([torch.nn.Embedding(151, 256) for _ in range(mixing_num)])
        self.h = torch.nn.ModuleList([GPT2Block(config, layer_idx=i) if i <= 11 else GiLTGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_relpos = None
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        attn_relpos = torch.clip(attn_relpos, 0, 150).long()
        attn_relpos = torch.cat([self.rel_embed[i](attn_relpos[i]) for i in range(mixing_num)], dim=-1) # batch, l, l, 256 * 3
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            if attention_mask is not None:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                if i <= 11:
                    outputs = block(
                        hidden_states,
                        layer_past=layer_past,
                        attention_mask=attention_mask,
                        head_mask=head_mask[i],
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )
                else:
                    outputs = block(
                        hidden_states,
                        layer_past=layer_past,
                        attention_mask=attention_mask,
                        head_mask=head_mask[i],
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        attn_relpos=attn_relpos
                    )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class GiLTGPT2LMHead(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GiLTGPT2(config)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        arrows = None
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # process arrows -> attn_relpos here, we will use self.rel_embed, attn_relpos (mixing num, b, l, l)
        left_sents_arrow, right_sents_arrow = arrows
        sents_index_to_id = synchronize_arrows(input_ids)
        length_i = input_ids.shape[1]
        degree_relpos = []
        degree_relpos_for_pointer = []
        depth_relpos = []
        depth_relpos_for_pointer = []
        distance_relpos = []
        distance_relpos_for_pointer = []
        for left_sent_arrow, right_sent_arrow, sent_index_to_id in zip(left_sents_arrow, right_sents_arrow, sents_index_to_id):
            input_size = len(sent_index_to_id) - 1
            id_to_index = {}
            for i in range(len(sent_index_to_id)):
                if sent_index_to_id[i] != -1:
                    if sent_index_to_id[i] not in id_to_index:
                        id_to_index[sent_index_to_id[i]] = [i]
                    else:
                        id_to_index[sent_index_to_id[i]].append(i)
            
            graph_len = len(left_sent_arrow) + 1
            graph_distance = np.zeros((graph_len, graph_len))
            graph = np.zeros((graph_len, graph_len))
            for i in range(len(left_sent_arrow)):
                if left_sent_arrow[i]:  #i <- j
                    for j in left_sent_arrow[i]:
                        graph[j][i + 1] = 1
                        graph[i + 1][j] = 1
                        graph_distance[j][i + 1] = 10 # 10
                        graph_distance[i + 1][j] = 1
                if right_sent_arrow[i]: #i -> j
                    for j in right_sent_arrow[i]:
                        graph[i + 1][j] = 1
                        graph[j][i + 1] = 1 
                        graph_distance[i + 1][j] = 10 # 10
                        graph_distance[j][i + 1] = 1
            sent_degree_relpos = np.zeros((length_i, length_i))
            sent_degree_relpos_step = np.zeros((length_i))
            sent_degree_relpos_for_pointer = np.zeros((max(sent_index_to_id) + 1, (max(sent_index_to_id)) + 1))
            sent_depth_relpos = np.zeros((length_i, length_i))
            sent_depth_relpos_for_pointer = np.zeros((max(sent_index_to_id) + 1, (max(sent_index_to_id)) + 1))
            sent_depth_relpos_for_pointer[:, 0] = 1
            sent_distance_relpos = np.zeros((length_i, length_i))
            sent_distance_relpos_for_pointer = np.zeros((max(sent_index_to_id) + 1, (max(sent_index_to_id)) + 1))
            finished_word_idx = -1
            degree_for_pointer = np.zeros((max(sent_index_to_id) + 1))
            for i in range(input_size):
                if sent_index_to_id[i] == -1:
                    # if finetune is not None:
                    #     sent_degree_relpos_step = np.zeros((length_i))
                    #     degree_for_pointer = np.zeros((max(sent_index_to_id) + 1))
                    continue
                tmp_finished_word_idx = finished_word_idx
                finished_word_idx = sent_index_to_id[i] # first token i - 1
                depth = calculate_depth(graph[:finished_word_idx + 1, :finished_word_idx + 1])
                for id, depth_value in enumerate(depth[1:]):
                    sent_depth_relpos[i, id_to_index[id + 1]] = depth_value
                for id, depth_value in enumerate(depth):
                    sent_depth_relpos_for_pointer[finished_word_idx, id] = depth_value
                distance = dijkstra(graph_distance[:finished_word_idx + 1, :finished_word_idx + 1], finished_word_idx)
                for id, distance_value in enumerate(distance[1:]):
                    sent_distance_relpos[i, id_to_index[id + 1]] = distance_value
                for id, distance_value in enumerate(distance):
                    sent_distance_relpos_for_pointer[finished_word_idx, id] = distance_value
                if finished_word_idx != tmp_finished_word_idx:
                    add_arc_idx = finished_word_idx
                    if left_sent_arrow[add_arc_idx - 1]:  #i <- j
                        for j in left_sent_arrow[add_arc_idx - 1]:
                            if j != 0:
                                sent_degree_relpos_step[id_to_index[j]] += 10 # 10
                            sent_degree_relpos_step[id_to_index[add_arc_idx]] += 1
                            degree_for_pointer[j] += 10 # 10
                            degree_for_pointer[add_arc_idx] += 1

                    if right_sent_arrow[add_arc_idx - 1]: #i -> j
                        for j in right_sent_arrow[add_arc_idx - 1]:
                            if j != 0:
                                sent_degree_relpos_step[id_to_index[j]] += 1
                            sent_degree_relpos_step[id_to_index[add_arc_idx]] += 10 # 10
                            degree_for_pointer[j] += 1
                            degree_for_pointer[add_arc_idx] += 10 # 10
                sent_degree_relpos[i] = copy.deepcopy(sent_degree_relpos_step)
                sent_degree_relpos_for_pointer[add_arc_idx] = copy.deepcopy(degree_for_pointer)
            degree_relpos_for_pointer.append(sent_degree_relpos_for_pointer[:-1])
            degree_relpos.append(sent_degree_relpos)
            depth_relpos_for_pointer.append(sent_depth_relpos_for_pointer[:-1])
            depth_relpos.append(sent_depth_relpos)
            distance_relpos_for_pointer.append(sent_distance_relpos_for_pointer[:-1])
            distance_relpos.append(sent_distance_relpos)
        
        batch = input_ids.shape[0]
        attn_relpos = torch.LongTensor(np.array([degree_relpos, depth_relpos, distance_relpos])).to(input_ids.device)   # 3, batch, l, l
        # attn_relpos = attn_relpos.view(batch, length_i, length_i, -1)
        attn_relpos_for_pointer = degree_relpos_for_pointer + depth_relpos_for_pointer + distance_relpos_for_pointer
        attn_relpos_for_pointer = [torch.LongTensor(np.array([attn_relpos_for_pointer[i + j * batch] for j in range(mixing_num)])).to(input_ids.device) for i in range(batch)]
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attn_relpos=attn_relpos
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        ), attn_relpos_for_pointer, sents_index_to_id
    
    def generate(self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_relpos = None):
        # attn_relpos is (3,b,1,l) l is length up to now
        attn_relpos = torch.clip(attn_relpos, 0, 150)
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attn_relpos=attn_relpos
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

def eval(model, eval_data, is_GiLT=False, left_arrow=None, right_arrow=None, biaffine_model=None):
    model.eval()
    if biaffine_model is not None:
        biaffine_model.eval()
    total_loss = 0.0
    total_num = 0
    crit = torch.nn.BCELoss(reduction="none")
    crit2 = torch.nn.CrossEntropyLoss(reduction="none")
    arc_acc = 0
    action_num = 0
    topk_infer_num = 0
    topk_acc_num = 0
    label_num = 0
    cs_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    for idx, eval_ids in enumerate(eval_data):
        max_len = max([len(eval_id) for eval_id in eval_ids])
        sum_len = sum([len(eval_id) for eval_id in eval_ids])
        eval_inps = torch.ones((len(eval_ids), max_len - 1), dtype=torch.long) * 0
        eval_tgts = torch.ones((len(eval_ids), max_len - 1), dtype=torch.long) * -1
        for i, eval_id in enumerate(eval_ids):
            eval_id = torch.tensor(eval_id, dtype=torch.long)
            eval_inps[i, :len(eval_id) - 1] = eval_id[:-1]
            eval_tgts[i, :len(eval_id) - 1] = eval_id[1:]
        eval_inps = eval_inps.to(device)
        eval_tgts = eval_tgts.to(device)
        if not is_GiLT:
            outputs = model(eval_inps)
            loss = cs_loss(outputs.logits.permute(0,2,1), eval_tgts).sum()
        else:
            eval_arrows = [left_arrow[idx], right_arrow[idx]]
            outputs, attn_relpos_for_pointer, sents_index_to_id = model(eval_inps, arrows=eval_arrows, output_hidden_states=True)
            hidden_states = torch.cat([outputs.hidden_states[-2], outputs.hidden_states[13]], dim=-1)
            emb = outputs.hidden_states[0]
            batch_predicates_input = predicate_alignment(hidden_states, sents_index_to_id, emb)

            biaffine_loss = []
            if [left_arrow[idx], right_arrow[idx]]:
                for sent_index_to_id, predicates_input, sent_left_label, sent_right_label, attn_rel in zip(
                    sents_index_to_id, batch_predicates_input, left_arrow[idx], right_arrow[idx], attn_relpos_for_pointer):
                    max_word_length = max(sent_index_to_id)
                    labels = torch.zeros(max_word_length + 1, max_word_length + 1)
                    arc_num_labels = torch.zeros(max_word_length)
                    for index, (left_label, right_label) in enumerate(zip(sent_left_label, sent_right_label)):
                        labels[index + 1, right_label] = 1
                        labels[left_label, index + 1] = 1
                        arc_num_labels[index] = len(left_label) + len(right_label)

                    if predicates_input:
                        labels = (labels[:,1:]).unsqueeze(0).to(device)
                        arc_num_labels = arc_num_labels.unsqueeze(0).long().to(device)
                        scores, arc_prob, arc_logits = biaffine_model(torch.stack(predicates_input).unsqueeze(0), attn_rel.unsqueeze(1))
                        loss = crit(scores, labels).sum() + crit2(arc_logits.permute(0,2,1), arc_num_labels).sum()
                        biaffine_loss.append(loss)

                        arc_num_prediction = torch.argmax(arc_prob, dim=-1)
                        arc_scores = scores.clone()
                        for index, (arc_pred, arc_true) in enumerate(zip(arc_num_prediction[0], arc_num_labels[0])):
                            if arc_pred == arc_true:
                                arc_acc += 1
                            topk_infer_num += arc_pred
                            temp_scores = arc_scores[:, :(index+2), :(index+1)]
                            topk_values, topk_indices = torch.topk(temp_scores.flatten(), k=min(arc_pred, 2 * (index + 1)))
                            rows = topk_indices // temp_scores.shape[2]
                            cols = topk_indices % temp_scores.shape[2]
                            arc_scores[:, :(index+2), :(index+1)] = 0
                            for row, col in zip(rows, cols):
                                if labels[0, row, col] == 1:
                                    topk_acc_num += 1
                            
                        action_num += len(arc_num_labels[0])   
                        label_nonzero = torch.nonzero(labels, as_tuple=False)
                        label_num += len(label_nonzero)
            loss = cs_loss(outputs.logits.permute(0,2,1), eval_tgts).sum() + torch.stack(biaffine_loss).sum()
        # index = [len(eval_idx) - 3 for eval_idx in eval_ids]
        # prediction = torch.argmax(outputs.logits, dim=-1)[torch.arange(len(eval_ids)), index]
        # acc_num = (prediction == eval_tgts[torch.arange(len(eval_ids)), index]).sum().item()
        total_num += sum_len
        total_loss += loss.item()

    if topk_infer_num != 0 and is_GiLT:
        topk_pre = topk_acc_num / topk_infer_num
        topk_recall = topk_acc_num / label_num
        if topk_pre == 0 or topk_recall == 0:
            topk_f1 = 0
        else:
            topk_f1 = 2 * topk_pre * topk_recall / (topk_pre + topk_recall)
    else:
        topk_f1 = 0
        topk_pre = 0
        topk_recall = 0
    logger.info(f"topk pre {topk_pre:.4f}, topk rec {topk_recall:.4f}, topk f1 {topk_f1:.4f}")
    logger.info(f"action num acc {arc_acc / action_num:.4f}")
    model.train()
    if biaffine_model is not None:
        biaffine_model.train()
    return np.exp(total_loss / total_num)

def create_scheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

if __name__ == "__main__":
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    is_GiLT = True
    accumulate_step = 4
    train_bz = 64 // accumulate_step if is_GiLT else 64
    train_path = "../data_process/GPT2-tokenizer/BLLIP_LG_TRAIN.txt"    # retokenize it, shuffle 不对齐
    test_path = "../data_process/GPT2-tokenizer/BLLIP_LG_TEST.txt"
    dev_path = "../data_process/GPT2-tokenizer/BLLIP_LG_DEV.txt"
    train_arrow_path = "../data_process/GPT2-tokenizer/BLLIP_GPT2_TRAIN_psd_multiarrow.txt"
    dev_arrow_path = "../data_process/GPT2-tokenizer/BLLIP_GPT2_DEV_psd_multiarrow.txt"
    test_arrow_path = "../data_process/GPT2-tokenizer/BLLIP_GPT2_TEST_psd_multiarrow.txt"

    train_data = load_data(train_path, batchsize=train_bz, seed=seed, shuffle=True, size="demo")
    test_data = load_data(test_path, batchsize=8, seed=seed, shuffle=False)
    dev_data = load_data(dev_path, batchsize=8, seed=seed, shuffle=False)

    EPOCHS = 2
    LEARNING_RATE = 5e-5    # 3e-5
    WEIGHT_DECAY = 1e-4    # 1e-4
    total_steps = len(train_data) * EPOCHS // accumulate_step if is_GiLT else len(train_data) * EPOCHS
    warmup_steps = 3000

    tokenizer = GPT2Tokenizer.from_pretrained("/home/huangty/GPT2/medium355M")
    tokenizer.add_prefix_space = True
    if not is_GiLT:
        model = GPT2LMHeadModel.from_pretrained("/home/huangty/GPT2/medium355M") # output_hidden_states=True
    else:
        model = GiLTGPT2LMHead.from_pretrained("/home/huangty/GPT2/medium355M")
        biaffine_model = BiaffineAttention(4096, 1024, type="Multi")
        left_train_arrow, right_train_arrow = load_multiarrow(train_arrow_path, batchsize=train_bz, shuffle=True, seed=seed, size="demo")
        left_dev_arrow, right_dev_arrow = load_multiarrow(dev_arrow_path, batchsize=8, shuffle=False, seed=seed)
        left_test_arrow, right_test_arrow = load_multiarrow(test_arrow_path, batchsize=8, shuffle=False, seed=seed)
        biaffine_model = biaffine_model.to(device)
        biaffine_model.train()
        biaffine_optimizer = torch.optim.AdamW(biaffine_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        biaffine_scheduler = create_scheduler(biaffine_optimizer, warmup_steps, total_steps)
    model = model.to(device)
    eos_string = tokenizer.eos_token
    bos_string = tokenizer.bos_token
    pad_id = 50256
    
    configure_logger("logs/gpt2_post_training.log")
    logger = get_logger()
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Weight decay: {WEIGHT_DECAY}")
    save_path = os.path.join("models/GiLT_gpt2_medium_post.pt")
    biaffine_save_path = os.path.join("models/GiLT_gpt2_biaffine.pt")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = create_scheduler(optimizer, warmup_steps, total_steps)
    if is_GiLT:
        optimizer.zero_grad()
        biaffine_optimizer.zero_grad()
    cs_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    crit = torch.nn.BCELoss(reduction="none")
    crit2 = torch.nn.CrossEntropyLoss(reduction="none")

    best_ppl = 1000000
    best_test_ppl = 0
    step_count = 0
    sum_loss = 0.0
    sum_words = 0

    log_step = 20 * (accumulate_step if is_GiLT else 1)
    eval_step = 500 * (accumulate_step if is_GiLT else 1)

    early_stop_signal = 0
    for epoch in range(EPOCHS):
        if early_stop_signal >= 3:
            break
        logger.info('=' * 30 + f"EPOCH {epoch + 1} started" + '=' * 30)
        for idx, train_ids in enumerate(train_data):
            max_len = max([len(train_id) for train_id in train_ids])
            sum_len = sum([len(train_id) for train_id in train_ids])
            train_inps = torch.ones((len(train_ids), max_len - 1), dtype=torch.long) * 0
            train_tgts = torch.ones((len(train_ids), max_len - 1), dtype=torch.long) * -1
            for i, train_id in enumerate(train_ids):
                train_id = torch.tensor(train_id, dtype=torch.long)
                train_inps[i, :len(train_id) - 1] = train_id[:-1]
                train_tgts[i, :len(train_id) - 1] = train_id[1:]
            train_inps = train_inps.to(device)
            train_tgts = train_tgts.to(device)
            
            if not is_GiLT:
                outputs = model(train_inps)
                loss = cs_loss(outputs.logits.permute(0,2,1), train_tgts).sum()
                total_loss = loss
            else:
                train_arrows = [left_train_arrow[idx], right_train_arrow[idx]]
                outputs, attn_relpos_for_pointer, sents_index_to_id = model(train_inps, arrows=train_arrows, output_hidden_states=True)
                # import pdb;pdb.set_trace()
                # out2 = model.generate(train_inps[:,:1], attn_relpos=torch.zeros(3,16,1,1).long(), use_cache=True, return_dict=True, output_hidden_states=True)
                hidden_states = torch.cat([outputs.hidden_states[-2], outputs.hidden_states[13]], dim=-1)
                emb = outputs.hidden_states[0]
                batch_predicates_input = predicate_alignment(hidden_states, sents_index_to_id, emb)

                biaffine_loss = []
                if [left_train_arrow[idx], right_train_arrow[idx]]:
                    for sent_index_to_id, predicates_input, sent_left_label, sent_right_label, attn_rel in zip(
                        sents_index_to_id, batch_predicates_input, left_train_arrow[idx], right_train_arrow[idx], attn_relpos_for_pointer):
                        max_word_length = max(sent_index_to_id)
                        labels = torch.zeros(max_word_length + 1, max_word_length + 1)
                        arc_num_labels = torch.zeros(max_word_length)
                        for idx, (left_label, right_label) in enumerate(zip(sent_left_label, sent_right_label)):
                            labels[idx + 1, right_label] = 1
                            labels[left_label, idx + 1] = 1
                            arc_num_labels[idx] = len(left_label) + len(right_label)

                        if predicates_input:
                            labels = (labels[:,1:]).unsqueeze(0).to(device)
                            arc_num_labels = arc_num_labels.unsqueeze(0).long().to(device)
                            scores, arc_prob, arc_logits = biaffine_model(torch.stack(predicates_input).unsqueeze(0), attn_rel.unsqueeze(1))
                            loss = crit(scores, labels).sum() + crit2(arc_logits.permute(0,2,1), arc_num_labels).sum()
                            biaffine_loss.append(loss)
                loss = 1/6 * torch.stack(biaffine_loss).sum() + 5/6 * cs_loss(outputs.logits.permute(0,2,1), train_tgts).sum()
                total_loss = torch.stack(biaffine_loss).sum() + cs_loss(outputs.logits.permute(0,2,1), train_tgts).sum()
            # index = [len(train_idx) - 3 for train_idx in train_idxs]
            # loss = loss[torch.arange(len(train_idxs)), index].mean()

            if is_GiLT:
                loss.backward()
                if (step_count + 1) % accumulate_step == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                    torch.nn.utils.clip_grad_norm_(biaffine_model.parameters(), 3.0)
                    biaffine_optimizer.step()
                    biaffine_scheduler.step()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    biaffine_optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                scheduler.step()
            sum_loss += total_loss.item()
            sum_words += sum_len

            step_count += 1
            if step_count % log_step == 0:
                mean_loss = sum_loss / sum_words
                logger.info(f"Epoch {epoch+1} Step {step_count} / {len(train_data) * EPOCHS}, LR {optimizer.param_groups[0]['lr']:.7f}, loss {mean_loss:.4f}, ppl {np.exp(mean_loss):.4f}")
                sum_loss = 0.0
                sum_words = 0
            
            if step_count % eval_step == 0 or step_count == len(train_data) * EPOCHS:
                # test on dev
                if not is_GiLT:
                    dev_ppl = eval(model, dev_data, is_GiLT)
                else:
                    dev_ppl = eval(model, dev_data, is_GiLT, left_dev_arrow, right_dev_arrow, biaffine_model)
                logger.info(f"Dev PPL: {dev_ppl}")
                if dev_ppl < best_ppl:
                    early_stop_signal = 0
                    best_ppl = dev_ppl
                    if not is_GiLT:
                        test_ppl = eval(model, test_data, is_GiLT)
                    else:
                        test_ppl = eval(model, test_data, is_GiLT, left_test_arrow, right_test_arrow, biaffine_model)
                        torch.save(biaffine_model.state_dict(), biaffine_save_path)
                    best_test_ppl = test_ppl
                    logger.info(f"Test PPL: {test_ppl}, best so far.")
                    torch.save(model.state_dict(), save_path)
                else:
                    early_stop_signal += 1
                
                if early_stop_signal >= 3:
                    break
    
    logger.info(f"Best PPL: {best_ppl}")
    logger.info(f"best test PPL: {test_ppl}")
    logger.info(f"New model saved at {save_path}")
