# Rethinking the Necessity of Adaptive Retrieval-Augmented Generation through the Lens of Adaptive Listwise Ranking

This repository contains the cleaned code for the paper **Rethinking the Necessity of Adaptive Retrieval-Augmented Generation through the Lens of Adaptive Listwise Ranking**.

Our work revisits a central assumption behind adaptive retrieval-augmented generation: better answer generation may not always require a fully adaptive generation mechanism. Instead, a strong document selection stage can already recover much of the benefit. To study this question, we compare standard retrieval-based generation, choose-based generation, and rerank-based generation under a unified evaluation pipeline.

The experiments are conducted on three benchmarks with different answer styles and evaluation requirements:
- `ASQA`: ambiguous long-form question answering
- `ELI5`: explanatory long-form generation
- `QAMPARI`: list-style answer generation

## Method Perspective

The repository is organized around three experimental settings used in the paper:
- `Vanilla`: generate directly from top retrieved documents
- `Choose`: generate from a selected subset of candidate documents
- `Rerank`: generate from documents reordered by a reranker

The core perspective of the paper is that the `Choose` stage should be viewed as **adaptive listwise ranking over a candidate set**. In our experiments, the chooser operates over a fixed document pool and selects subsets such as top-`1`, `3`, `5`, and `10` documents before answer generation.

This means the released code is centered on the interface between:
- retrieval outputs
- choose/rerank outputs
- generation
- evaluation

rather than on a single monolithic adaptive-RAG framework.

## Repository Structure

```text
ijcnn2026/
├── config_paths.py
├── generation/
│   ├── vanilla/run_vanilla.py
│   ├── choose/run_choose.py
│   └── rerank/run_rerank.py
├── evaluation/
│   ├── local/eval.py
│   ├── local/utils.py
│   ├── vanilla/batch_eval_vanilla.py
│   ├── choose/batch_eval_choose.py
│   └── rerank/batch_eval_rerank.py
├── training/
│   └── axolotl_template.yml
└── docs/
    ├── training.md
    └── training_data_format.md
```

## Main Components

### Generation
- `generation/vanilla/run_vanilla.py`
  Runs answer generation from vanilla retrieval results.
- `generation/choose/run_choose.py`
  Runs answer generation from precomputed choose-result directories configured in `config_paths.py`.
- `generation/rerank/run_rerank.py`
  Runs answer generation from reranked document lists.

### Evaluation
- `evaluation/local/eval.py`
  Main evaluator used by the repository. It contains the dataset-specific evaluation logic.
- `evaluation/vanilla/batch_eval_vanilla.py`
  Batch evaluation entry for vanilla generation results.
- `evaluation/choose/batch_eval_choose.py`
  Batch evaluation entry for choose-based generation results.
- `evaluation/rerank/batch_eval_rerank.py`
  Batch evaluation entry for rerank-based generation results.

### Training Notes
- `docs/training.md`
  Documents the missing training pipeline, the likely training-data structure, and how the released repository interfaces with an Axolotl-trained ranking model.
- `docs/training_data_format.md`
  Describes the expected structure of reconstructed training data for the choose-stage ranking model.
- `training/axolotl_template.yml`
  A reconstruction-oriented Axolotl template config. This is not claimed to be the exact training config used in the paper.

## Dataset-Specific Differences

Although the three experimental settings share the same overall pipeline, the benchmarks differ in prompt design and evaluation.

### Generation differences
- `ASQA` uses long-form answer prompts.
- `ELI5` uses explanatory answer prompts with a concise journalistic style.
- `QAMPARI` uses list-style prompts and expects comma-separated answers.

### Evaluation differences
- `ASQA` mainly uses string-based answer coverage and ROUGE-style metrics.
- `ELI5` uses claim-based evaluation.
- `QAMPARI` uses list precision, recall, and F1.

## Configuration

All external resource paths are centralized in `config_paths.py`.
The file is organized into four sections:
- `GENERATOR_MODELS`: generation model checkpoints
- `EVALUATION_MODELS`: evaluator checkpoints used by `evaluation/local/eval.py`
- `VANILLA_INPUTS`, `CHOOSE_INPUTS`, `RERANK_INPUTS`: experiment inputs
- `RESULT_DIRS`: output directories for generated answers

For `Choose`, `config_paths.py` supports multiple input directories through `CHOOSE_INPUTS["source_dirs"]`. This keeps the choose pipeline independent of any single chooser model name while preserving compatibility with different precomputed choose outputs.

If you move models or result folders, update `config_paths.py` instead of editing each script separately.

## Training

The ranking model used for the `Choose` stage was trained with **Axolotl**. The original training scripts are no longer available in the repository, so this release does not claim exact training-script reproducibility.

To make the repository complete and honest, we provide:
- a training note in `docs/training.md`
- a training-data format note in `docs/training_data_format.md`
- a reconstruction-oriented Axolotl template in `training/axolotl_template.yml`

The training note explains the likely structure of the training data: each instance pairs a question with a candidate document list and supervision for listwise selection, so that the trained model can export choose-result files compatible with the released pipeline.

## Inputs And Outputs

This repository does not bundle large datasets, checkpoints, or full experiment outputs.

The code expects the following kinds of inputs:
- vanilla retrieval files
- choose result files
- rerank result files
- local or shared model checkpoints

Generated outputs are written to repository-level result directories such as:
- `results_vanilla_qwen3`
- `results_choose_qwen3`
- `results_rerank_qwen3_8B`

## Running The Code

### Run generation
```bash
python generation/vanilla/run_vanilla.py
python generation/choose/run_choose.py
python generation/rerank/run_rerank.py
```

### Run evaluation
```bash
python evaluation/vanilla/batch_eval_vanilla.py
python evaluation/choose/batch_eval_choose.py
python evaluation/rerank/batch_eval_rerank.py
```

### Run the evaluator directly
```bash
python evaluation/local/eval.py --f path/to/result.jsonl
```

## Evaluation Arguments

The local evaluator supports the following command-line arguments:
- `--f`: input result file in JSONL format
- `--output_column`: column name for model outputs, default is `output`
- `--no_rouge`: disable ROUGE evaluation
- `--qa`: enable QA-based evaluation
- `--mauve`: enable MAUVE evaluation
- `--citations`: enable citation evaluation
- `--at_most_citations`: maximum number of cited documents used in citation evaluation
- `--claims_nli`: enable claim-based evaluation for `ELI5`
- `--cot`: for `QAMPARI`, split chain-of-thought and answer list when needed

Typical examples:

```bash
python evaluation/local/eval.py --f path/to/result.jsonl
python evaluation/local/eval.py --f path/to/eli5_result.jsonl --claims_nli
python evaluation/local/eval.py --f path/to/result.jsonl --citations
python evaluation/local/eval.py --f path/to/qampari_result.jsonl --cot
```

The batch evaluation scripts already add the dataset-specific flags they need. For example, `ELI5` evaluation automatically includes `--claims_nli`.

## Notes

- The repository is a cleaned experiment-facing version of the original workspace rather than a newly redesigned framework.
- All required code dependencies are included inside `ijcnn2026`.
- Remaining external dependencies are limited to models and experiment input/output files configured in `config_paths.py`.
