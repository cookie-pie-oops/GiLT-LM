# beam search: we want to model p(y|x) here, where y{1:n} is the tree (the attachment decisions) and x{1:n} is the sequence.
# x{1:n} is given for every step from 1 to n.

import numpy as np
from heapq import heapify, heappush, heappop, heappushpop
from typing import List, Tuple
import torch
from copy import deepcopy
from model_bllip_con import PushdownTransformerConstituency

class BeamObj:
    def __init__(self, score, score_seq, attachment_decisions, step):
        self.score = score # score of the whole (partial) sequence = log p(x,y)_{1:i}
        self.score_seq = score_seq # score seq of every step [log p(x_j,y_j)]_{1:i}
        self.attachment_decisions = attachment_decisions # list of attachment decisions at every step [a_j]_{1:i}
        self.step = step # the step of the beam object, i.e. the number of attachment decisions made so far = i
        # (actually i-1 because we start from 0)


def logsumexp(x):
    x = np.array(x)
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

class BeamSearchDepthBased:
    """
    Beam Search given the whole sequence of attachment decisions.
    P(y_i | x_{1:n})
    """
    def __init__(self, beam_size=300):
        self.beam_size = beam_size
        
    def update_beam(self,
                    ids,
                    model: PushdownTransformerConstituency,
                    beam_curr: List[BeamObj],
                    list_reduced,
                    stack_tape
                    ) -> List[BeamObj]:
        """
        Update the beam with new candidates.
        """
        assert ids[0] == 1, "Beam search only supports batch size of 1 for now."
        
        # id batch size = id[0]
        # beam batch size = [beam_size] !!!
        beam_next = [] # [beam_size]
        step = beam_curr[0].step + 1 # we want to predict the next step
        # ids expand to [beam_size, seqlen]
        ids = ids.repeat(self.beam_size, 1)
        with torch.no_grad():
            # get the scores for the new candidates
            # ids: [beam_size, seqlen]
            # reduced_set: [beam_size,] and every element is a set of reduced attachment decisions
            # stack_tape: [beam_size, step, step]
            # model: a function that takes in ids and returns scores
            
            # log prob. so we need to find MAXIMUM.
            scores_word, scores_attach_time = model.take_step_silver_tree(ids, stack_tape, list_reduced, step)
            # scores_word: [beam_size,]
            # scores_attach_time: [beam_size, step+1]
        # update the beam
        if step != ids[0].shape[1]-1: # -> we are not at the end of the sequence
            seen_preds = set()
            heapify(beam_next)
            for i, beam_obj in enumerate(beam_curr):
                # get the new score
                new_score = beam_obj.score + scores_word[i].item()
                # get the new attachment decisions
                for atta_idx, atta_score in enumerate(scores_attach_time[i]):
                    if atta_score.item() == float("-inf"):
                        continue
                    if atta_idx == 0:
                        # if the attachment decision is 0, that's the whole sequence,
                        # and we will handle this elsewhere
                        continue
                    new_score += atta_score.item()
                    # get the new attachment decisions
                    new_attachment_decisions = beam_obj.attachment_decisions + [atta_idx]
                    # check if we have seen this attachment decision before
                    if tuple(new_attachment_decisions) in seen_preds:
                        continue
                    seen_preds.add(tuple(new_attachment_decisions))
                    # create a new beam object
                    new_beam_obj = BeamObj(new_score, 
                                        beam_obj.score_seq + [
                                            (
                                                scores_word[i].item(), 
                                                scores_attach_time[i][atta_idx].item()
                                            )
                                        ],
                                        new_attachment_decisions, 
                                        step)
                    # add the new beam object to the heap
                    if len(beam_next) < self.beam_size:
                        heappush(beam_next, (-new_score, new_beam_obj)) # because we want to maximize the score,
                        # we need to push the negative score to mimic a max heap with a min heap
                    else:
                        heappushpop(beam_next, (-new_score, new_beam_obj))
        elif step == ids[0].shape[1]-1: # -> we are at the end of the sequence
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
                                        step)
                if len(beam_next) < self.beam_size:
                    heappush(beam_next, (-new_score, new_beam_obj))
                else:
                    heappushpop(beam_next, (-new_score, new_beam_obj))
        else:
            raise ValueError("Invalid step!")
        real_beam_next = [beam_obj for _, beam_obj in beam_next]
        assert len(real_beam_next) <= self.beam_size, f"Beam size exceeded: {len(real_beam_next)} > {self.beam_size}"
        return real_beam_next
    
    def init_beam(self):
        return BeamObj(
            score=0.0,
            score_seq=[],
            attachment_decisions=[],
            step=0
        )

    def _no_reduce_op(self, stack_pred, step_prime):
        return stack_pred == step_prime # +1 is deleted because step_prime starts from 1.
    
    def _update_reduced_states(self, reduced_state, stack_pred, step_prime):
        ### if stack_pred != step_prime = last_step + 1, then everything from stack_pred to step is reduced
        for elem in range(stack_pred, step_prime):
            reduced_state.add(elem)
            
    def _update_stacks(self, reduced_states, stacks, attachment_decisions, step_prime, depths):
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
            if self._no_reduce_op(stack_pred, step_prime):
                stack.append([step_prime])
            else:
                self._update_reduced_states(reduced_states[idx], stack_pred, step_prime)
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
                ):
        assert ids[0] == 1, "Beam search only supports batch size of 1 for now."
        beam_curr = [self.init_beam() for _ in range(self.beam_size)]
        # beam_curr: [beam_size]
        stack_tape = np.zeros((self.beam_size, ids[0].shape[1], ids[0].shape[1]), dtype=np.int64)
        stack_tape_one_row = np.zeros((self.beam_size, ids[0].shape[1]), dtype=np.int64)
        list_reduced = [set() for _ in range(self.beam_size)]
        stacks_history = [[[0]] for _ in range(self.beam_size)]
        
        for step_prime in range(1, ids[0].shape[1]):
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
            list_reduced = [
                deepcopy(list_reduced[i]) for i in range(self.beam_size)
            ]
            # stack_tape = np.stack([
            #     deepcopy(stack_tape[i]) for i in range(self.beam_size)
            # ]) # FIXME: WHY??
            
            
            stacks_history = [
                deepcopy(stacks_history[i]) for i in range(self.beam_size)
            ]
            
            # update stacks and things
            stacks_history, list_reduced, stack_tape_one_row = self._update_stacks(
                reduced_states=list_reduced,
                stacks=stacks_history,
                attachment_decisions=attach_decision_recent,
                step_prime=step_prime,
                depths=stack_tape_one_row
            )
            stack_tape[:, step_prime, :] = stack_tape_one_row
            
        # return: beams
        # marginal log prob = logsumexp([b.score for b in beam_curr])
        marginal_log_prob = logsumexp([b.score for b in beam_curr]) # log p(x)_{1:n} where we want to know -1/N(all sents) * sum(marginal of all sents)

        # i.e. TODO: MARGINAL PPL = exp(-1/sum(lengths)*sum(marginal log prob))
        return beam_curr, marginal_log_prob

        
        