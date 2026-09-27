import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import torch
from strong_reject.evaluate import evaluate as sr_evaluate
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from prefix import test_prefixes
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run io eval.")
    parser.add_argument("--goal-target-path", type=Path, required=True)
    parser.add_argument("--prompt-audio-path", type=Path, required=True)
    parser.add_argument("--output-dir", default="result")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=-1)
    return parser.parse_args()


def strongreject_score(prompt: str, response: str) -> float:
    try:
        return sr_evaluate(prompt, response, ["strongreject_rubric"])[0]["score"]
    except Exception:
        return 0.0


def build_model_and_tokenizer(model_path: str):
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        fix_mistral_regex=True,
    )

    print("[INFO] Loading model")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("[INFO] Model loaded successfully")
    print("[INFO] Model device map:", model.hf_device_map)
    return model, tokenizer


def get_model_response(model, tokenizer, audio_path: str, prompt: str, generation_config: dict) -> str:
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    with torch.no_grad():
        try:
            response = model.chat(tokenizer, messages, generation_config=generation_config)
        except TypeError:
            response = model.chat(tokenizer, generation_config, messages)

    return response


def main(args: argparse.Namespace) -> None:
    goal_target_path = Path(args.goal_target_path)
    prompt_audio_path = Path(args.prompt_audio_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (args.output_file or f"io.json")

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False
    }

    model, tokenizer = build_model_and_tokenizer(args.model_path)

    with open(goal_target_path, "r", encoding="utf-8") as f:
        goals = json.load(f)
    with open(prompt_audio_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    total_available = min(len(goals), len(prompts))
    if args.max_examples is None or args.max_examples <= 0:
        total = total_available
    else:
        total = min(total_available, args.max_examples)

    print(f"[INFO] Total available samples: {total_available}", flush=True)
    print(f"[INFO] Selected samples: {total}", flush=True)
    print(f"[INFO] Output -> {output_file}", flush=True)
    
    results = []

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[\n")

    for i in tqdm(range(total), desc="Running Omni", ncols=80):
        goal_item = goals[i]
        prompt_item = prompts[i]

        goal = goal_item.get("goal")
        jailbreak_prompt = prompt_item.get("text_input")
        audio_file = prompt_item.get("audio_input")

        try:
            response = get_model_response(model, tokenizer, audio_file, jailbreak_prompt, generation_config)
        except Exception as e:
            print(f"[ERROR] index {i} inference failed: {e}", flush=True)
            response = f"[ERROR_OMNI_LOCAL] {e}"

        score = strongreject_score(jailbreak_prompt, response)

        one_result = {
            "goal": goal,
            "jailbreak_prompt": jailbreak_prompt,
            "audio_file": audio_file,
            "audio_response": response,
            "audio_score": score,
        }
        
        results.append(one_result)

        with open(output_file, "a", encoding="utf-8") as f:
            comma = "," if i < total - 1 else ""
            f.write(json.dumps(one_result, ensure_ascii=False, indent=2))
            f.write(comma)
            f.write("\n")

    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n]")

    print(f"[INFO] Done. Results written to: {output_file}", flush=True)

    count = 0
    total = 0
    for item in results:
        total += 1
        response = item['audio_response']
        if not any([prefix in response for prefix in test_prefixes]):
            count += 1

    print("="*20)
    print(f"Total: {total}, Count: {count}, KeyWords-ASR: {count / total}")
    print("="*20)

    avg_text_score = sum(item.get("audio_score", 0) for item in results) / len(results)
    print("average strongreject score is ", avg_text_score)

if __name__ == "__main__":
    main(parse_args())
