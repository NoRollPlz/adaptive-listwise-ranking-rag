import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
EVAL_SCRIPT = REPO_ROOT / 'evaluation' / 'local' / 'eval.py'
RESULT_DIR = REPO_ROOT / 'results_choose_qwen3'


def detect_dataset(name: str):
    lower = name.lower()
    if 'asqa' in lower:
        return 'asqa', []
    if 'qampari' in lower:
        return 'qampari', []
    if 'eli5' in lower:
        return 'eli5', ['--claims_nli']
    return None, None


def run_eval(file_path: Path, extra_args):
    print('=' * 80)
    print(f'[EVAL] {file_path}')
    print('=' * 80)
    cmd = [PYTHON, str(EVAL_SCRIPT), '--f', str(file_path)] + extra_args
    subprocess.run(cmd, check=True)


def main():
    if not EVAL_SCRIPT.exists():
        print(f'ERROR: {EVAL_SCRIPT} not found')
        return

    if not RESULT_DIR.exists():
        print(f'[SKIP] {RESULT_DIR} not found')
        return

    found = False
    for path in sorted(RESULT_DIR.glob('*.jsonl')):
        dataset, extra_args = detect_dataset(path.name)
        if dataset is None:
            print(f'[SKIP] Unrecognized dataset for {path.name}')
            continue
        found = True
        run_eval(path, extra_args)

    if not found:
        print('No choose result files found.')
        return

    print('\nChoose evaluations finished.')


if __name__ == '__main__':
    main()
