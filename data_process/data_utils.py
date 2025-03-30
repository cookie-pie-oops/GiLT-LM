from nltk.tree import Tree

def flatten(parse, add_eos):
    def helper(p):
        if type(p) == str:
            return p
        else:
            return " ".join(helper(x) for x in p)

    if type(parse) == Tree:
        words = " ".join(parse.leaves())
    else:
        words = helper(parse)
    if add_eos:
        return "{} </s>".format(words)
    else:
        return words
    
def binarize_tree(parse):
    if type(parse) == str:
        return parse
    else:
        if len(parse) == 1:
            return binarize_tree(parse[0])
        else:
            return (binarize_tree(parse[0]), binarize_tree(parse[1:]))

def post_process_op(tree, sp):
    """
    Input: nltk tree where leaf nodes are strings
    Output: nltk tree but use spm tokenizer to tokenize leaf nodes and create a subtree corresponding to it.
    """

    def fix(t, is_first, label=None):
        if type(t) == str:
            # tokenized = tokenizer.tokenize(t, add_prefix_space=not is_first)
            tokenized = sp.encode_as_pieces(t)
            if len(tokenized) == 1:
                return Tree(label, [tokenized[0]])
            else:
                return Tree(label, [Tree(label, [tok]) for tok in tokenized])
        elif len(t) == 1:
            return fix(t[0], is_first=is_first, label=t.label())
        else:
            return Tree(
                t.label(),
                [fix(c, is_first=is_first and idx == 0) for idx, c in enumerate(t)],
            )

    fixed_tree = fix(tree, is_first=True)
    fixed_tree.chomsky_normal_form()
    return fixed_tree