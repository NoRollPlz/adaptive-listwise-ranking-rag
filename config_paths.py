from pathlib import Path

# Repository root. All relative input/output paths are resolved from here.
REPO_ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Model checkpoints
# Update these paths if you use different local or shared model directories.
# -----------------------------------------------------------------------------
GENERATOR_MODELS = {
    "default": "/zhdd/home/zche/models/Qwen3-8B",
}

EVALUATION_MODELS = {
    "qa": "/data/models/roberta-large-squad",
    "autoais": "/data/models/t5_xxl_true_nli_mixture",
    "mauve": "/data/models/gpt2-large",
}

# -----------------------------------------------------------------------------
# Experiment inputs
# These paths point to retrieval files or precomputed choose/rerank results.
# For choose, multiple source directories can be configured here.
# -----------------------------------------------------------------------------
VANILLA_INPUTS = {
    "asqa": REPO_ROOT / "data" / "vanilla_inputs" / "asqa_eval_gtr_top100.json",
    "qampari": REPO_ROOT / "data" / "vanilla_inputs" / "qampari_eval_gtr_top100.json",
    "eli5": REPO_ROOT / "data" / "vanilla_inputs" / "eli5_eval_bm25_top100.json",
}

CHOOSE_INPUTS = {
    "source_dirs": [
        REPO_ROOT / "gpt4_choose_result",
        Path("/data/hlv/ourmodel/choose_result") / "4_augment_fix",
    ],
    "default_files": {
        "asqa": "asqa_eval_gtr_top100_choose_result.jsonl",
        "eli5": "eli5_eval_bm25_top100_choose_result.jsonl",
        "qampari": "qampari_eval_gtr_top100_choose_result.jsonl",
    },
}

RERANK_INPUTS = {
    "results_dir": REPO_ROOT / "rank_result",
    "datasets": {
        "asqa": "asqa_gpt-4_top10_rerank.jsonl",
        "qampari": "qampari_gpt-4_top10_rerank.jsonl",
        "eli5": "eli5_gpt-4_top10_rerank.jsonl",
    },
}

# -----------------------------------------------------------------------------
# Output directories
# Generated answers and evaluation-ready result files are written here.
# -----------------------------------------------------------------------------
RESULT_DIRS = {
    "vanilla": REPO_ROOT / "results_vanilla_qwen3",
    "choose": REPO_ROOT / "results_choose_qwen3",
    "rerank": REPO_ROOT / "results_rerank_qwen3_8B",
}
