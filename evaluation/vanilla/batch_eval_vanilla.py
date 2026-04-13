import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "evaluation" / "local" / "eval.py"
RESULT_DIR = REPO_ROOT / "results_vanilla_qwen3"

K_LIST = [5]
MODES = ["nonthinking", "thinking"]
DATASETS = {
    "asqa": {"extra_args": []},
    "qampari": {"extra_args": []},
    "eli5": {"extra_args": ["--claims_nli"]},
}

for dataset, cfg in DATASETS.items():
    for k in K_LIST:
        for mode in MODES:
            file_path = RESULT_DIR / f"{dataset}_qwen3_vanilla_{k}_{mode}.jsonl"
            if not file_path.is_file():
                print(f"[SKIP] {file_path} not found")
                continue

            print("=" * 80)
            print(f"[EVAL] {dataset.upper()} | vanilla-{k} | {mode}")
            print(f"FILE: {file_path}")
            print("=" * 80)

            cmd = [PYTHON, str(EVAL_SCRIPT), "--f", str(file_path)] + cfg["extra_args"]
            subprocess.run(cmd, check=True)

print("All vanilla evaluations (ASQA / QAMPARI / ELI5) finished.")
