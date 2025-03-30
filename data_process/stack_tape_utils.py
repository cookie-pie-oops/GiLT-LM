import numpy as np

def compute_stack_tape(
    attachment_labels,
    head_info,
    type_labels=None,
    with_depth_info=False,
):
    ### the stack_tape is a matrix of size len(attachment_labels) x len(attachment_labels) where the (i, j) entry is 1 if
    ### after observing the input string till position i-1, the jth token has either participated in a reduce operation.
    ### given the stack labels for an input string, compute the penalty matrices for the input string
    ### for example, if the input_str is "a b c d" and the parse is "((a b) (c d))", then the stack labels are
    ### [0, 0, 2, 1] and the penalty matrix is [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]

    num_words = len(attachment_labels)
    penalty_matrix = np.zeros((num_words, num_words))
    curr_state = np.zeros(num_words)

    ### simulate the shift reduce operations to get depth info
    if with_depth_info:
        depth = np.zeros(num_words)
        stack = []

    for i in range(num_words):
        if with_depth_info:
            penalty_matrix[i] = depth
        else:
            penalty_matrix[i] = curr_state
        ### update the stack and depths
        if with_depth_info:
            if attachment_labels[i] == i:
                stack.append([i])
            else:
                ### this means that the ith token has participated in a reduce operation, so we do reduces.
                curr_constituent = [i]
                while len(stack) > 1 and attachment_labels[i] not in stack[-1]:
                    top = stack.pop()
                    curr_constituent = top + curr_constituent
                    ### update depth
                    for c in curr_constituent:
                        depth[c] += 1

                top = stack.pop()
                curr_constituent = top + curr_constituent
                ### update depth
                for c in curr_constituent:
                    depth[c] += 1
                stack.append(curr_constituent)
        ### update curr_state info
        ### if attachment_labels[i] != i, then the ith token has participated in a reduce operation
        if attachment_labels[i] != i:
            if i == num_words - 1:
                start = attachment_labels[i]
            else:
                start = head_info[i - 1] + 1
            curr_state[start : i + 1] = 1 if type_labels is None else type_labels[i]
        elif type_labels is not None:
            curr_state[i : i + 1] = type_labels[i]
    return np.concatenate([penalty_matrix[1:], penalty_matrix[-1][None, :]], axis=0)