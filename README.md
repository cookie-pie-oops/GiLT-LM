# SDP_for_TG

## Introduction

Our code is based on ["Dependency Transformer Grammars: Integrating Dependency Structures into Transformer Language Models"](https://arxiv.org/abs/2407.17406v1). After finishing experiments, we will perfect the code reposity.

## Environment

You can follow `./Dep_Transformer_Grammars/environments.yml` for reference environment and read the README.md in `./Dep_Transformer_Grammars` for more details about the environment.

## Data process

We split the BLLIP corpus into train, dev and test sets with `./data_process/BLLIP_process.py` following the DTG as well as TG.

If you have a SDP graph, transfer it into the format given below in a json file:

{
    "text": "We use ...",
    "psd_graph": [
            {
                "id": "1",
                "form": "We",
                "head":[
                    head_id,
                    ...
                ]
            },
            ...
            ],
    "dm_graph": [...],
    "pas_graph": [...]
}

Then use `./data_process/add_arc_to_corpus.py` to generate two edge label file as the input of training part.

## Training

We use `./Dep_Transformer_Grammars/train_graphLayer.sh` to start training. It require a token level file and two edge label files.

The evaluate step in training only evaluate the loss of transformer neglecting the biaffine model, we need evaluate after totally training to obtain the performance.

## Evaluation

We use `./Dep_Transformer_Grammars/eval.sh` to start evaluation, it require a token level file and two edge label files as well as the trained model path.

The ppl given include the biaffine model and transformer. To find out the tighter upper bound of ppl, we sample 300 parse graph with `./data_process/sample_parse_graph.py` and set the eval_type=estimate in `./Dep_Transformer_Grammars/eval.sh`.