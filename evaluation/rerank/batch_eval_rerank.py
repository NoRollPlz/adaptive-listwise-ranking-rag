import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "evaluation" / "local" / "eval.py"
RESULT_DIR = REPO_ROOT / "results_rerank_qwen3_8B"

DATASETS = {
    "asqa": [],
    "qampari": [],
    "eli5": ["--claims_nli"],
}
K_LIST = [1, 3, 5, 10]
MODES = ["nonthinking", "thinking"]

for dataset, extra in DATASETS.items():
    for k in K_LIST:
        for mode in MODES:
            file_path = RESULT_DIR / f"{dataset}_qwen3_rerank_{k}_{mode}.jsonl"
            if not file_path.exists():
                print(f"[SKIP] {file_path}")
                continue

            print(f"\n[EVAL] {dataset} | rerank | top-{k} | {mode}")
            cmd = [PYTHON, str(EVAL_SCRIPT), "--f", str(file_path)] + extra
            subprocess.run(cmd, check=True)

print("All rerank evaluations finished.")
