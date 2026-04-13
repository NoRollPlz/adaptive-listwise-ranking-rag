# Training Data Format

This document describes the expected structure of reconstructed training data for the adaptive listwise ranking model.

Because the original training-data construction code is not available, this format description is intended to document the interface implied by the released repository rather than claim the exact original internal format.

## Task View

Each training example corresponds to:
- one question
- one candidate document list
- one supervision signal describing which documents should be preferred, selected, or ordered

The downstream `Choose` pipeline assumes that the trained model produces a selection over a fixed candidate pool and exports the result as a JSONL record.

## Core Fields

A reconstructed record should contain at least the following fields:
- `question`: the input question
- `docs`: a list of candidate documents
- one selection field among `chosen_docs`, `choose_order`, or `selected_ids`

### `question`
The user query or benchmark question.

Example:
```json
"question": "Who wrote The Old Man and the Sea?"
```

### `docs`
A list of candidate documents retrieved before the `Choose` stage.
Each document is typically represented as:
- `title`
- `text`

Example:
```json
"docs": [
  {"title": "Ernest Hemingway", "text": "Ernest Miller Hemingway was an American novelist..."},
  {"title": "The Old Man and the Sea", "text": "The Old Man and the Sea is a short novel written by..."}
]
```

## Selection Supervision

The released pipeline supports three interchangeable ways of representing the chooser output.

### `selected_ids`
A list of zero-based indices into `docs`.

Example:
```json
"selected_ids": [1, 0]
```

### `choose_order`
A textual ordering over the candidate list.
The released code can parse index-like patterns from this field.

Example:
```json
"choose_order": "[2] > [1] > [3]"
```

### `chosen_docs`
A materialized list of selected document objects.

Example:
```json
"chosen_docs": [
  {"title": "The Old Man and the Sea", "text": "The Old Man and the Sea is a short novel written by..."}
]
```

## Optional Benchmark Fields

The released generation and evaluation code will preserve benchmark-specific fields when available, including:
- `qa_pairs`
- `annotations`
- `answers`
- `answer`
- `claims`

These fields are not necessary for the chooser itself, but they are useful for downstream evaluation and result packaging.

## Minimal Record Example

```json
{
  "question": "Who wrote The Old Man and the Sea?",
  "docs": [
    {"title": "Ernest Hemingway", "text": "Ernest Miller Hemingway was an American novelist..."},
    {"title": "The Old Man and the Sea", "text": "The Old Man and the Sea is a short novel written by Ernest Hemingway in 1951..."},
    {"title": "John Steinbeck", "text": "John Steinbeck was an American author..."}
  ],
  "selected_ids": [1, 0],
  "choose_order": "[2] > [1] > [3]"
}
```

## Relation To The Paper Setting

For the paper experiments, the `Choose` stage should be understood as adaptive listwise ranking over a candidate set rather than binary independent relevance scoring.
In practice, this means:
- the model observes a ranked or unordered candidate pool
- the supervision reflects relative preference among candidates
- the exported output can later be truncated to top-`1`, `3`, `5`, or `10`

## Export To Inference

After training, the ranking model should generate choose-result JSONL files that can be consumed directly by:
- `generation/choose/run_choose.py`
- `evaluation/choose/batch_eval_choose.py`

So long as the exported file preserves the required interface fields, it can be plugged into the released repository without changing the downstream code.
