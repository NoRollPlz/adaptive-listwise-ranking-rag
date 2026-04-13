import json
import sys
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config_paths import GENERATOR_MODELS, RESULT_DIRS, VANILLA_INPUTS

QWEN_MODEL_PATH = GENERATOR_MODELS["default"]
OUTPUT_DIR = RESULT_DIRS["vanilla"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VANILLA_K_LIST = [5]
THINKING_MODES = ["non-thinking", "thinking"]
DATASET_CONFIGS = {dataset: {"input": path} for dataset, path in VANILLA_INPUTS.items()}


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list, but got {type(data)}")
    return data


def select_docs_vanilla(docs, k):
    return [] if k == 0 else docs[:k]


def extract_final_answer(text):
    if not isinstance(text, str):
        return str(text)
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def build_prompt(tokenizer, question, docs, dataset_name, thinking=False):
    content = ""
    if docs:
        for i, d in enumerate(docs):
            content += f"[{i+1}] {d.get('title','')}: {d.get('text','')}\n\n"

    if dataset_name == "asqa":
        content += "\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.\n"
    elif dataset_name == "eli5":
        if docs:
            content += "\nWrite an accurate, engaging, and concise answer for the given question using only the provided search results. Use an unbiased and journalistic tone. \n"
        else:
            content += "\nWrite an accurate, engaging, and concise answer for the given question. Use an unbiased and journalistic tone. \n"
    elif dataset_name == "qampari":
        if docs:
            content += "\nProvide a list of accurate answers for the given question using only the provided search results. Separate answers by commas. For questions that have more than 5 answers, write at least 5 answers.\n"
        else:
            content += "\nProvide a list of accurate answers for the given question. Separate answers by commas. For questions that have more than 5 answers, write at least 5 answers.\n"

    content += f"Question: {question}.\nAnswer:"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def run_generation(dataset_name, dataset_cfg, k, thinking, llm):
    print(f"\nRunning {dataset_name} | vanilla-{k} | {'thinking' if thinking else 'non-thinking'}")
    tokenizer = llm.get_tokenizer()
    data = load_data(dataset_cfg["input"])

    prompts = []
    meta = []
    for item in data:
        question = item["question"]
        docs = item.get("docs", [])
        selected_docs = select_docs_vanilla(docs, k)
        prompts.append(build_prompt(tokenizer, question, selected_docs, dataset_name, thinking))
        meta.append({
            "question": question,
            "qa_pairs": item.get("qa_pairs"),
            "annotations": item.get("annotations"),
            "answers": item.get("answers"),
            "answer": item.get("answer"),
            "claims": item.get("claims"),
            "docs": selected_docs,
        })

    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096 if thinking else 1024)
    outputs = llm.generate(prompts, sampling_params)

    suffix = "thinking" if thinking else "nonthinking"
    out_path = OUTPUT_DIR / f"{dataset_name}_qwen3_vanilla_{k}_{suffix}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for out, item in zip(outputs, meta):
            answer = extract_final_answer(out.outputs[0].text)
            result = {
                "question": item["question"],
                "output": answer,
                "qa_pairs": item["qa_pairs"],
                "annotations": item["annotations"],
                "answers": item["answers"],
                "answer": item.get("answer"),
                "claims": item["claims"],
                "docs": item["docs"],
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Saved to {out_path}")


def main():
    print("Loading Qwen3-8B with vLLM...")
    llm = LLM(model=QWEN_MODEL_PATH, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
    for dataset_name, dataset_cfg in DATASET_CONFIGS.items():
        for k in VANILLA_K_LIST:
            for mode in THINKING_MODES:
                run_generation(dataset_name, dataset_cfg, k, thinking=(mode == "thinking"), llm=llm)


if __name__ == "__main__":
    main()
