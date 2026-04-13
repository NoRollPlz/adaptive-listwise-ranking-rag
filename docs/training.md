# Training Note

The original training scripts for the adaptive listwise ranking model used in the paper are no longer available in this repository. However, the model itself was trained with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl).

This document is therefore a reproduction-oriented note rather than an exact release of the original training pipeline.

## What We Can Confirm

- The ranking model used in the paper was trained with Axolotl.
- The released repository focuses on inference, generation, and evaluation.
- The exact original training launcher and data-preparation code were not preserved.

## Intended Role Of The Training Model

In this project, the training model is used for the `Choose` stage.
Its role is to score or rank candidate documents and produce a choose result, which is then consumed by:
- `generation/choose/run_choose.py`
- `evaluation/choose/batch_eval_choose.py`

The downstream generation code assumes that the choose stage has already produced structured choose-result files.

## Training Data

Although the original data-construction code is missing, the released pipeline and the paper setting make the training data format reasonably clear.

A detailed format description is provided in:
- `docs/training_data_format.md`

In short, each reconstructed training example should pair:
- a `question`
- a candidate document list `docs`
- a supervision signal such as `chosen_docs`, `choose_order`, or `selected_ids`

## Recommended Reproduction Strategy

If you want to reconstruct the training pipeline, the most practical route is:
1. Prepare supervised data for document ranking/selection.
2. Fine-tune the ranking model with Axolotl.
3. Export choose-result files in the format expected by the generation pipeline.
4. Run generation and evaluation using the released scripts.

## Axolotl Template

This repository includes a minimal template config at:
- `training/axolotl_template.yml`

It is only a starting point for reconstruction. It is **not** claimed to be the exact camera-ready training configuration.

## Important Limitation

Because the original training code is missing, this repository does **not** claim exact training reproducibility for the ranking model. Instead, it documents:
- the framework used for training: Axolotl
- the role of the trained model in the pipeline
- the likely structure of the training data
- the interface between training outputs and released inference/evaluation code

If exact training reproduction becomes necessary later, the missing parts that still need to be recreated are:
- data construction for ranking supervision
- the exact Axolotl config used in the paper
- launch commands and hardware settings
- checkpoint selection and post-processing details
