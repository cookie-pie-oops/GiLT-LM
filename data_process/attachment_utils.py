from nltk.tree import Tree

def accomodate_sos_and_eos(attachment_labels, constituent_labels=None):
    ### the first 0 is for the SOS token and the last 0 is for the EOS token
    if not constituent_labels:
        return [0] + [1 + label for label in attachment_labels] + [0], None
    else:
        return [0] + [1 + label for label in attachment_labels] + [0], ["SOS"] + [
            label for label in constituent_labels
        ] + ["EOS"]

def get_shift_reduce_actions(parse):
    """
    given an input_str and its corresponding parse as a tuple of tuples, return the series of shift reduce operations that would produce the parse
    Examples:
    if the input_str is "a b c d" and the parse is "((a, b), (c, d))", then the shift reduce operations are
    shift a, shift b, reduce, shift c, shift d, reduce, reduce
    if the input_str is "a b c d" and the parse is "(a, (b, (c, d)))", then the shift reduce operations are
    shift a, shift b, shift c, shift d, reduce reduce reduce
    if the input_str is "a b c d" and the parse is "(((a, b), c), d)", then the shift reduce operations are
    shift a, shift b, reduce, shift c, reduce, shift d, reduce
    """

    actions = []
    # parse leaves length
    # print("parse leaves length:", len(parse.leaves()))
    def shift_reduce_recursive(parse):
        if type(parse) == str or len(parse) == 1:
            actions.append("shift")
        else:
            for p in parse:
                shift_reduce_recursive(p)
            actions.append("reduce")

    shift_reduce_recursive(parse)
    # print("Actions:", actions)
    return actions

def get_constituent_labels(tree_tuple):
    split_vals = {}

    def get_len(t, st, label=None):
        if type(t) == str:
            split_vals[(st, st)] = label
            return 1
        elif len(t) == 1:
            return get_len(t[0], st, label=t.label())
        else:
            curr_len = 0
            for c in t:
                l1_len = get_len(c, st + curr_len)
                curr_len += l1_len
            split_vals[(st, st + curr_len - 1)] = t.label()
            return curr_len

    get_len(tree_tuple, 0)
    return split_vals

def compute_attachment_labels_text(parse, input_pieces, with_depth_info=True):
    """
    given an input_str and its corresponding parse, return for each token in the input_str the earliest token
    in the input_str that the token wants to reduce with.
    Examples:
    if the input_str is "a b c d" and the parse is "((a b) (c d))", then the stack labels are
    [0, 0, 2, 1] because the series of shift reduce operations that would produce the parse is
    shift a (0), shift b (1), reduce (2), shift c (3), shift d (4), reduce (5), reduce (6)
    if the input_str is "a b c d" and the parse is "(a, (b, (c, d)))", then the stack labels are
    [0, 1, 2, 0] because the series of shift reduce operations that would produce the parse is
    shift a (0), shift b (1), shift c (2), shift d (3), reduce (4),  reduce (5), reduce (6)
    if the input_str is "a b c d" and the parse is "(((a, b), c), d)", then the stack labels are
    [0, 0, 1, 2] because the series of shift reduce operations that would produce the parse is
    shift a (0), shift b (1), reduce (2), shift c (3), reduce (4), shift d (5), reduce (6)

    if with_depth_info, then along with type labels, return the number of reduce operations that have been performed.
    """

    stack_actions = get_shift_reduce_actions(parse)

    if type(parse) == Tree:
        constituent_labels_given = get_constituent_labels(parse)
        constituent_labels = []
        # print(constituent_labels_given)
        # print(parse)
        # print(input_str)
        
    attachment_labels = []
    stack = []

    curr_word_idx = 0
    stack_idx = 0
    cstart = {}

    if "</s>" in input_pieces:
        ### for reasons, we added a eos token to the input string
        # num_words = len(input_str.split(" ")) - 1
        num_words = len(input_pieces) - 1  #### remove one for the EOS token!!!!
    else:
        # num_words = len(input_str.split(" "))
        num_words = len(input_pieces)
    while curr_word_idx < num_words:
        next_shift = stack_idx + 1

        stack_top = None
        while next_shift < len(stack_actions) and stack_actions[next_shift] != "shift":
            stack_top = stack.pop()
            next_shift += 1

        if stack_top is None:
            attachment_labels.append(curr_word_idx)
            constituent_labels.append(
                constituent_labels_given[(curr_word_idx, curr_word_idx)]
            )
        else:
            ### everything from head[stack_top] to curr_word_idx is a constituent.
            ### also get the constituent label for the constituent
            attachment_labels.append(stack_top)
            if stack_top not in cstart:
                constituent_start = stack_top
            else:
                constituent_start = cstart[stack_top]
            constituent_labels.append(
                constituent_labels_given[(constituent_start, curr_word_idx)]
            )

            cstart[curr_word_idx] = constituent_start

        stack.append(curr_word_idx)
        stack_idx = next_shift
        curr_word_idx += 1

    attachment_labels, type_labels = accomodate_sos_and_eos(
        attachment_labels, constituent_labels
    )

    if type_labels is not None:
        return {
            "type_labels": type_labels,
            "attachment_labels": attachment_labels,
            "cstart_info": cstart,
        }
    else:
        return {
            "attachment_labels": attachment_labels,
            "cstart_info": cstart,
        }
