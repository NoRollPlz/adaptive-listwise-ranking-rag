import json
import re
import sys
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config_paths import CHOOSE_INPUTS, GENERATOR_MODELS, RESULT_DIRS

QWEN_MODEL_PATH = GENERATOR_MODELS["default"]
CHOOSE_RESULT_DIRS = CHOOSE_INPUTS["source_dirs"]
DEFAULT_FILES = CHOOSE_INPUTS["default_files"]
OUTPUT_DIR = RESULT_DIRS["choose"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    data = []
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def select_chosen_docs(item):
    if isinstance(item.get("chosen_docs"), list):
        return item["chosen_docs"]

    docs_pool = item.get("docs", [])
    if not isinstance(docs_pool, list) or not docs_pool:
        return []

    if isinstance(item.get("choose_order"), str):
        indices = re.findall(r"\d+", item["choose_order"])
        selected_docs = []
        for idx_str in indices:
            try:
                idx = int(idx_str) - 1
            except ValueError:
                continue
            if 0 <= idx < len(docs_pool):
                doc = docs_pool[idx]
                if isinstance(doc, dict):
                    selected_docs.append(doc)
        return selected_docs

    if isinstance(item.get("selected_ids"), list):
        selected_docs = []
        for idx in item["selected_ids"]:
            if isinstance(idx, int) and 0 <= idx < len(docs_pool):
                doc = docs_pool[idx]
                if isinstance(doc, dict):
                    selected_docs.append(doc)
        return selected_docs

    return []


def build_prompt(tokenizer, question, docs, mode="nonthinking", dataset=None):
    question = "" if question is None else str(question)
    content = ""

    if isinstance(docs, list) and docs:
        for i, doc in enumerate(docs):
            if not isinstance(doc, dict):
                continue
            title = str(doc.get("title", ""))
            text = str(doc.get("text", ""))
            content += f"[{i+1}] {title}: {text}\n\n"

    if dataset == "asqa":
        content += "\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.\n"
    elif dataset == "eli5":
        if docs:
            content += "\nWrite an accurate, engaging, and concise answer for the given question using only the provided search results. Use an unbiased and journalistic tone. \n"
        else:
            content += "\nWrite an accurate, engaging, and concise answer for the given question. Use an unbiased and journalistic tone. \n"
    elif dataset == "qampari":
        if docs:
            content += "\nProvide a list of accurate answers for the given question using only the provided search results. Separate answers by commas. For questions that have more than 5 answers, write at least 5 answers.\n"
        else:
            content += "\nProvide a list of accurate answers for the given question. Separate answers by commas. For questions that have more than 5 answers, write at least 5 answers.\n"

    if mode == "thinking":
        content += "\nPlease keep your internal thought process concise and under 1024 tokens.\n"

    content += f"Question: {question}.\nAnswer:"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=(mode == "thinking"),
    )


def extract_final_answer(text):
    if not isinstance(text, str):
        return str(text)
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def detect_dataset_type(name):
    lower = name.lower()
    if "asqa" in lower:
        return "asqa"
    if "eli5" in lower:
        return "eli5"
    if "qampari" in lower:
        return "qampari"
    return None


def iter_tasks(input_dir):
    tasks = []

    if input_dir.exists():
        for path in sorted(input_dir.iterdir()):
            if path.is_file() and path.suffix == ".jsonl":
                dataset = detect_dataset_type(path.name)
                if dataset:
                    tasks.append((dataset, path.name))

    if tasks:
        return tasks

    for dataset, filename in DEFAULT_FILES.items():
        tasks.append((dataset, filename))
    return tasks


def main():
    llm = LLM(
        model=QWEN_MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.37,
    )
    tokenizer = llm.get_tokenizer()
    modes = ["nonthinking", "thinking"]

    for source_index, input_dir in enumerate(CHOOSE_RESULT_DIRS, start=1):
        source_tag = f"source{source_index}"
        tasks = iter_tasks(input_dir)

        for dataset, filename in tasks:
            choose_path = input_dir / filename
            data = load_jsonl(choose_path)
            if not data:
                continue

            for mode in modes:
                print(f"\n[GEN] {dataset.upper()} | choose | {source_tag} | Qwen3-8B | {mode}")
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=4096 if mode == "thinking" else 1024,
                )
                prompts = []
                metas = []
                for item in data:
                    question = item.get("question", "")
                    docs = select_chosen_docs(item)
                    prompts.append(build_prompt(tokenizer, question, docs, mode, dataset))
                    metas.append({
                        "question": question,
                        "docs": docs,
                        "qa_pairs": item.get("qa_pairs"),
                        "annotations": item.get("annotations"),
                        "answers": item.get("answers"),
                        "answer": item.get("answer"),
                        "claims": item.get("claims"),
                        "source_tag": source_tag,
                        "source_dir": str(input_dir),
                        "source_file": filename,
                    })

                outputs = llm.generate(prompts, sampling_params)
                out_file = OUTPUT_DIR / f"{source_tag}_{Path(filename).stem}_qwen3_choose_{mode}.jsonl"
                with open(out_file, "w", encoding="utf-8") as f:
                    for out, meta in zip(outputs, metas):
                        answer = extract_final_answer(out.outputs[0].text)
                        record = {
                            "question": meta["question"],
                            "output": answer,
                            "docs": meta["docs"],
                            "qa_pairs": meta["qa_pairs"],
                            "annotations": meta["annotations"],
                            "answers": meta["answers"],
                            "answer": meta.get("answer"),
                            "claims": meta["claims"],
                            "source_tag": meta["source_tag"],
                            "source_dir": meta["source_dir"],
                            "source_file": meta["source_file"],
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"[SAVED] {out_file}")


if __name__ == "__main__":
    main()
