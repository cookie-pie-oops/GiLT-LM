# beam search: we want to model p(y|x) here, where y{1:n} is the tree (the attachment decisions) and x{1:n} is the sequence.
# x{1:n} is given for every step from 1 to n.

import numpy as np
from heapq import heapify, heappush, heappop, heappushpop
from typing import List, Tuple
import torch
from copy import deepcopy
from model_bllip_con import PushdownTransformerConstituency
import logging
import torch.nn as nn
class BeamObj:
    def __init__(self, score, score_seq, attachment_decisions, step):
        self.score = score # score of the whole (partial) sequence = log p(x,y)_{1:i}
        self.score_seq = score_seq # score seq of every step [log p(x_j,y_j)]_{1:i}
        self.attachment_decisions = attachment_decisions # list of attachment decisions at every step [a_j]_{1:i}
        self.step = step # the step of the beam object, i.e. the number of attachment decisions made so far = i
        # (actually i-1 because we start from 0)
    
    def __repr__(self):
        return f"BeamObj(score={self.score}, score_seq={self.score_seq}, attachment_decisions={self.attachment_decisions}, step={self.step})"
    
    def __str__(self):
        return self.__repr__()

    def __lt__(self, other):
        return self.score < other.score
    
    def __eq__(self, other):
        return self.score == other.score

def logsumexp(x):
    x = np.array(x)
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

# class TakeStepParallelWrapper(nn.Module):
#     def __init__(self, core_model: PushdownTransformerConstituency):
#         super().__init__()
#         self.core = core_model          # PushdownTransformerConstituency 实例

#     def forward(self, ids, stack_tape, reduced_mask, step):
#         # reduced_mask 必须是 [B, step+1] 的 bool Tensor
#         return self.core.take_step_silver_tree(
#             ids, stack_tape, reduced_mask, step
#         )

class BeamSearchDepthBased:
    """
    Beam Search given the whole sequence of attachment decisions.
    P(y_i | x_{1:n})
    """
    def __init__(self, beam_size=300, gold_attach=None):
        self.beam_size = beam_size

        if isinstance(gold_attach, torch.Tensor):
            gold_attach = gold_attach.tolist()
        self.gold_attach = gold_attach
    
        
    def update_beam(self,
                    ids,
                    model,
                    beam_curr: List[BeamObj],
                    list_reduced,
                    stack_tape
                    ) -> List[BeamObj]:
        """
        Update the beam with new candidates.
        
        Using a min-heap to keep track of the best beam_size candidates.
        Upon a new candidate is created, compare the score with the worst candidate in the heap.
        If the new candidate is better, replace the worst candidate with the new candidate.
        If the new candidate is worse, discard it.
        """
        # assert ids.shape[0] == 1, "Beam search only supports batch size of 1 for now."

        # id batch size = id[0]
        # beam batch size = [beam_size] !!!
        beam_next = [] # [beam_size]
        beam_next_heap = []
        now_step = beam_curr[0].step + 1 # we want to predict the next step
        # ids expand to [beam_size, seqlen]
        ids = ids.expand(self.beam_size, -1)
        seqlen = ids.shape[1]
        # beam_size = self.beam_size
        # if seqlen < self.beam_size:
        #     # use sort for n times will be O(n^2 log n)
        #     # use heap for n times will be O(n log n)
        
        with torch.no_grad():
            # get the scores for the new candidates
            # ids: [beam_size, seqlen]
            # reduced_set: [beam_size,] and every element is a set of reduced attachment decisions
            # stack_tape: [beam_size, step, step] 
            # model: a function that takes in ids and returns scores
            
            # log prob. so we need to find MAXIMUM.
            serial_step = 60
            # scores_word, scores_attach_time = model.take_step_silver_tree(ids, stack_tape, list_reduced, now_step)
            scores_word = []
            scores_attach_time = []
            for i in range(0, self.beam_size, serial_step):
                # get the scores for the new candidates
                # ids: [beam_size, seqlen]
                # reduced_set: [beam_size,] and every element is a set of reduced attachment decisions
                # stack_tape: [beam_size, step, step] 
                # model: a function that takes in ids and returns scores
                scores_word_tmp, scores_attach_time_tmp = model.take_step_silver_tree(
                    ids[i:i+serial_step], 
                    stack_tape[i:i+serial_step], 
                    list_reduced[i:i+serial_step],
                    now_step
                )
                scores_word.append(scores_word_tmp)
                scores_attach_time.append(scores_attach_time_tmp)
            scores_word = torch.cat(scores_word, dim=0)
            scores_attach_time = torch.cat(scores_attach_time, dim=0)
            # scores_word: [beam_size,]
            # scores_attach_time: [beam_size, step+1]
            
        # update the beam
        if now_step != seqlen-1: # -> we are not at the end of the sequence
            seen_preds = set()
            heapify(beam_next_heap)
            for i, beam_obj in enumerate(beam_curr):
                # get the new score
                new_score = beam_obj.score + scores_word[i].item()
                # get the new attachment decisions
                if False:
                    print("Current beam object: ", beam_obj)
                    print("For this attach, the scores: ", scores_attach_time[i])
                for atta_idx, atta_score in enumerate(scores_attach_time[i]):
                    if atta_score.item() == float("-inf"):
                        continue
                    if atta_idx == 0:
                        # if the attachment decision is 0, that's reducing with the whole sequence,
                        # and we will handle this elsewhere
                        continue
                    # new_score += atta_score.item()
                    
                    candidate_score = new_score + atta_score.item()
                    # get the new attachment decisions
                    new_attachment_decisions = beam_obj.attachment_decisions + [atta_idx]
                    # check if we have seen this attachment decision before
                    if tuple(new_attachment_decisions) in seen_preds:
                        continue
                    seen_preds.add(tuple(new_attachment_decisions))

                    # create a new beam object
                    new_beam_obj = BeamObj(candidate_score,
                                        beam_obj.score_seq + [
                                            (
                                                scores_word[i].item(), 
                                                # scores_attach_time[i][atta_idx].item()
                                                atta_score.item()
                                            )
                                        ],
                                        new_attachment_decisions, 
                                        now_step)
                    # add the new beam object to the heap
                    # NOTE: HEAP OPERATION -> TORCH TOPK?
                    # if len(beam_next_heap) < self.beam_size:
                    #     heappush(beam_next_heap, (candidate_score, new_beam_obj)) # because we want to maximize the score,
                    #     # we need to push the negative score to mimic a max heap with a min heap
                    # elif candidate_score > beam_next_heap[0][0]: # if the score is better and the heap is full, we need to pop the worst one
                    #     popped = heappushpop(beam_next_heap, (candidate_score, new_beam_obj))

                    beam_next.append(new_beam_obj)
        elif now_step == seqlen-1: # -> we are at the end of the sequence
            # with logprob = 0, i.e. prob = 1, we predict eos (or pad)
            for i, beam_obj in enumerate(beam_curr):
                eos_attach_score = scores_attach_time[i][0].item()
                new_score = beam_obj.score + scores_word[i].item() + eos_attach_score
                new_attachment_decisions = beam_obj.attachment_decisions + [0]
                new_beam_obj = BeamObj(new_score, 
                                        beam_obj.score_seq + [
                                            (
                                                scores_word[i].item(), 
                                                eos_attach_score
                                            )
                                        ],
                                        new_attachment_decisions,
                                        now_step)
                
                # NOTE: heap operation -> torch topk
                # if len(beam_next_heap) < self.beam_size:
                #     heappush(beam_next_heap, (new_score, new_beam_obj))
                # elif new_score > beam_next_heap[0][0]:
                #     popped = heappushpop(beam_next_heap, (new_score, new_beam_obj))
                beam_next.append(new_beam_obj)
        else:
            raise ValueError("Invalid step!")

        # torch topk
        if len(beam_next) > self.beam_size:
            _, topk_indices = torch.topk(torch.tensor([b.score for b in beam_next]), self.beam_size)
            topk_indices = topk_indices.tolist()
            real_beam_next = [beam_next[i] for i in topk_indices]
        else:
            real_beam_next = beam_next
        # assert len(real_beam_next) <= self.beam_size, f"Beam size exceeded: {len(real_beam_next)} > {self.beam_size}"
        # heap_scores = [b[0] for b in beam_next_heap]
        # true_scores = [b.score for b in real_beam_next]
        # print("heap scores logsumexp: ", logsumexp(bug_scores))
        # print("true scores logsumexp: ", logsumexp(true_scores))
        return real_beam_next
    
    def init_beam(self):
        return BeamObj(
            score=0.0,
            score_seq=[],
            attachment_decisions=[],
            step=0
        )
        
    @staticmethod
    def _no_reduce_op(stack_pred, step_prime):
        # +1 is deleted because step_prime starts from 1.
        return stack_pred == step_prime 
    
    @staticmethod
    def _update_reduced_states(reduced_state, stack_pred, step_prime):
        ### if stack_pred != step_prime = last_step + 1, then everything from stack_pred to step is reduced
        for elem in range(stack_pred, step_prime):
            reduced_state.add(elem)
    
    @staticmethod
    def _update_stacks(reduced_states, stacks, attachment_decisions, step_prime, depths):
        """
        Args:
            - reduced_states: a list of sets, each set contains the indices that are reduced
            - stacks: a list of lists, each list contains the indices that are on the stack
            - attachment_decisions: a list of ints, each int is the index of the token that step wants to reduce with
            - step: the current step
        Returns:
            - updated stacks, reduced_states and depths
        """

        ### this is the last step, we don't care about updating stacks at this point if we are synchronous
        if step_prime == len(depths[0]) - 1:
            return stacks, reduced_states, depths

        for idx, (stack, stack_pred) in enumerate(zip(stacks, attachment_decisions)):
            ### stack is a list of constituents, each constituent is a list of indices
            ### add [step] into stack_state
            if BeamSearchDepthBased._no_reduce_op(stack_pred, step_prime):
                stack.append([step_prime])
            else:
                BeamSearchDepthBased._update_reduced_states(reduced_states[idx], stack_pred, step_prime)
                curr_constituent = [step_prime]
                while len(stack) > 1 and stack_pred not in stack[-1]:
                    top = stack.pop()
                    curr_constituent = top + curr_constituent
                    for c in curr_constituent:
                        depths[idx][c] += 1
                top = stack.pop()
                curr_constituent = top + curr_constituent
                for c in curr_constituent:
                    depths[idx][c] += 1
                stack.append(curr_constituent)
        return stacks, reduced_states, depths 
    
    def __call__(self, 
                 model,
                 ids, # give bsz and seqlen, shape: [bsz, seqlen]
                 return_trail=False,
                ):
        
        # ids: bos, x_1, x_2, ..., x_n, eos
        # breakpoint()
        # gold_attach  shape: [bsz, seqlen]
        assert ids.shape[0] == 1, "Beam search only supports batch size of 1 for now."
        # logging.info("in")
        beam_curr = [self.init_beam() for _ in range(self.beam_size)]
        # beam_curr: [beam_size]
        seqlen = ids.shape[1]
        stack_tape = torch.zeros((self.beam_size, seqlen, seqlen), dtype=torch.long, device=ids.device)
        stack_tape_one_row = torch.zeros((self.beam_size, seqlen), dtype=torch.long, device=ids.device)
        list_reduced = [set() for _ in range(self.beam_size)]
        stacks_history = [[[0]] for _ in range(self.beam_size)]
        
        gold_not_found_step = None
        if return_trail:
            prefix_marginal_score_trajectory = []
        for step_prime in range(1, seqlen):
            # logging.info(f"Step: {step_prime}")
            # breakpoint()
            # step_prime is the step we are at, i.e. the number of attachment decisions made so far
            # ids: [bsz, seqlen]
            # reduced_set: [bsz, beam_size, max_stack_depth]
            # stack_tape: [bsz, beam_size, max_stack_depth]
            # model: a function that takes in ids and returns scores
            # stack_tape_tmp = stack_tape[:, :step_prime, :step_prime]
            beam_curr = self.update_beam(ids, model, beam_curr, list_reduced, stack_tape[:, :step_prime, :step_prime])

            # get attach decisions of all beams
            attach_decision_recent = [
                b.attachment_decisions[-1] for b in beam_curr
            ]
            # at_d = [b.attachment_decisions for b in beam_curr]
            # see if gold is in the beam
            # found = False
            # if self.gold_attach is not None:
            #     for i, b in enumerate(beam_curr):
            #         if self.gold_attach[0][:step_prime] == b.attachment_decisions:
            #             # logging.info(f"Gold attachment decision FOUND in beam {i}: {b.attachment_decisions}")
            #             # we can break here because we only need to find one
            #             found = True
            #             break
            # if not found and gold_not_found_step is None and self.gold_attach is not None:
            #     gold_not_found_step = step_prime
            #     logging.debug(f"Gold attachment decision started to be NOT found in beam: {self.gold_attach[0][:step_prime]}, step: {gold_not_found_step}, seqlen: {seqlen}")
            # breakpoint()
            
            list_reduced = deepcopy(list_reduced)
            stacks_history = deepcopy(stacks_history)
            
            # update stacks and things
            stacks_history, list_reduced, stack_tape_one_row = self._update_stacks(
                reduced_states=list_reduced,
                stacks=stacks_history,
                attachment_decisions=attach_decision_recent,
                step_prime=step_prime,
                depths=stack_tape_one_row
            )
            stack_tape[:, step_prime, :] = stack_tape_one_row.clone()
            
            # now compute a logsumexp, i.e. marginal log prob, then put into marginal_log_prob_trajectory
            if return_trail:
                prefix_marginal_score_trajectory.append(
                    torch.logsumexp(torch.tensor([b.score for b in beam_curr]), dim=0)
                )
            
        # return: beams
        # marginal log prob = logsumexp([b.score for b in beam_curr])
        # marginal_log_prob = logsumexp([b.score for b in beam_curr]) # log p(x)_{1:n} where we want to know -1/N(all sents) * sum(marginal of all sents)
        marginal_log_prob = torch.logsumexp(torch.tensor([b.score for b in beam_curr]), dim=0) # input shape [beam_size]
        # breakpoint()
        # i.e. TODO: MARGINAL PPL = exp(-1/sum(lengths)*sum(marginal log prob))
        
        if return_trail:
            return beam_curr, marginal_log_prob, prefix_marginal_score_trajectory
        else:
            return beam_curr, marginal_log_prob

        
        