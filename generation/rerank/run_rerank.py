import json
import re
import sys
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config_paths import GENERATOR_MODELS, RERANK_INPUTS, RESULT_DIRS

QWEN_MODEL_PATH = GENERATOR_MODELS["default"]
RANK_RESULT_DIR = RERANK_INPUTS["results_dir"]
OUTPUT_DIR = RESULT_DIRS["rerank"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
K_LIST = [1, 3, 5, 10]
DATASETS = RERANK_INPUTS["datasets"]


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def select_rerank_docs(item, k):
    docs = item.get("docs", [])
    if not docs:
        return []
    choose_order = item.get("choose_order")
    if isinstance(choose_order, str):
        indices = re.findall(r"\[(\d+)\]", choose_order)
        ordered_docs = []
        for idx_str in indices:
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < len(docs):
                    ordered_docs.append(docs[idx])
            except ValueError:
                continue
        if ordered_docs:
            return ordered_docs[:k]
    return docs[:k]


def build_prompt(tokenizer, question, docs, mode, dataset):
    content = ""
    if docs:
        for i, d in enumerate(docs):
            content += f"[{i+1}] {d.get('title', '')}: {d.get('text', '')}\n\n"

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
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=(mode == "thinking"))


def extract_final_answer(text):
    if not isinstance(text, str):
        return str(text)
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def main():
    llm = LLM(model=QWEN_MODEL_PATH, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.37)
    tokenizer = llm.get_tokenizer()
    modes = ["nonthinking", "thinking"]

    for dataset, fname in DATASETS.items():
        rank_path = RANK_RESULT_DIR / fname
        if not rank_path.exists():
            print(f"[WARN] File not found: {rank_path}, skipping...")
            continue
        data = load_jsonl(rank_path)

        for k in K_LIST:
            for mode in modes:
                print(f"\n[GEN] {dataset.upper()} | rerank @ {k} | Qwen3-8B | {mode}")
                sampling_params = SamplingParams(temperature=0.0, max_tokens=4096 if mode == "thinking" else 1024)
                prompts = []
                metas = []
                for item in data:
                    question = item["question"]
                    docs = select_rerank_docs(item, k)
                    prompts.append(build_prompt(tokenizer, question, docs, mode, dataset))
                    metas.append({
                        "question": question,
                        "docs": docs,
                        "qa_pairs": item.get("qa_pairs"),
                        "annotations": item.get("annotations"),
                        "answers": item.get("answers"),
                        "answer": item.get("answer"),
                        "claims": item.get("claims"),
                    })

                outputs = llm.generate(prompts, sampling_params)
                out_file = OUTPUT_DIR / f"{dataset}_qwen3_rerank_{k}_{mode}.jsonl"
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
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[SAVED] {out_file}")


if __name__ == "__main__":
    main()
