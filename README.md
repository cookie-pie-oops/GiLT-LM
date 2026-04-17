# [ACL 2026]GiLT-LM

## Introduction

Code Repository for our ACL 2026 paper "GiLT: Augmenting Transformer Language Models with Dependency Graphs"([paper](https://arxiv.org/abs/2605.15562)).

## Environment

You can follow `./environments.yml` for reference environment.

## Data process

We split the BLLIP corpus into train, dev and test sets with `./data_process/BLLIP_process.py` following the [DTG](https://github.com/zhaoyd1/Dep_Transformer_Grammars) as well as [TG](https://github.com/google-deepmind/transformer_grammars).

If you have a SDP graph, transfer it into the format given below in a json file:

```json
{
    "text": "We use ...",
    "psd_graph": [
        {
            "id": "1",
            "form": "We",
            "head": [
                "head id 1",
                "head id 2",
                "..."
            ]
        },
        {}
    ],
    "dm_graph": [],
    "pas_graph": []
}
```

Then use `./data_process/add_arc_to_corpus.py` to generate two edge label file as the input of training part.

## Before training

We collect each shell scripts into `./src/scripts`. But when doing experiments, we put it in `./src/*`. So before you start training or evaluating, make sure that the path is correct.

## Training

We use `./src/scripts/train_graphLayer.sh` to start training. It require a token level file and two edge label files.

The evaluate step in training only evaluate the loss of transformer neglecting the biaffine model, we need evaluate after totally training to obtain the performance.

## Evaluation

We use `./src/scripts/eval_BLLIP_beam_search.sh` to obtain the perplexity by beam search, it require a token level file and as well as the trained model path.

To obtain the SG score and BLiMP score. You should download the test set first and correct the path of test set (line 147 in `./src/SG_test_graphlayer.py`, line 76 in `./src/BLiMP_graphlayer.py`). Then use `./src/scripts/SG_test.sh` and `./src/scripts/BLiMP_test.sh` to get the score, it require the trained model path.

## Finetune

We use `./src/scripts/gpt2_post.sh` to start post-training leading to GiLT-GPT2 & Post-GPT2.

To finetune GiLT-GPT2 on the downstream task, we need parse the dataset with `./src/scripts/gpt2_parse.sh`

Since we use the GPT2 tokenizer, we have added the prompt format of RTE, SST2, STS-B and MRPC in the `./src/gpt2_finetune.py`, you can strightly finetune Post-GPT2 with a trained model file and GiLT-GPT2 with two extra parsed edge label files.

To obatin the score of GLUE tasks, our scripts will generate the answer of given input test csv, you should upload it to the GLUE.

To test the SG and BLiMP of GiLT-GPT2, you should check the annotated code in `./src/SG_test_graphlayer.py` and `./src/BLiMP_graphlayer.py`