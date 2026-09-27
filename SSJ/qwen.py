import argparse
import base64
import json
import time
from pathlib import Path
from prefix import test_prefixes
import requests
from strong_reject.evaluate import evaluate as sr_evaluate



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run remote Qwen Omni audio eval.")
    parser.add_argument("--goal-target-path", type=Path, required=True)
    parser.add_argument("--prompt-audio-path", type=Path, required=True)
    parser.add_argument("--output-dir", default="result")
    parser.add_argument("--output-file", default=None, help="Optional explicit output filename.")
    parser.add_argument("--model-name", default="qwen2.5-omni-7B", required=True)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0)
    return parser.parse_args()


def call_vllm(
    audio_path: str,
    prompt: str,
    api_url: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
) -> str:
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    }

    resp = requests.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    j = resp.json()

    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(j, ensure_ascii=False)


def strongreject_score(prompt: str, response: str) -> float:
    try:
        return sr_evaluate(prompt, response, ["strongreject_rubric"])[0]["score"]
    except Exception:
        return 0.0


def format_seconds(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def main(args: argparse.Namespace) -> None:
    model_name = args.model_name
    api_url = args.api_url

    goal_target_path = Path(args.goal_target_path)
    prompt_audio_path = Path(args.prompt_audio_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (args.output_file or f"{model_name}.json")

    with open(goal_target_path, "r", encoding="utf-8") as f:
        goals = json.load(f)
    with open(prompt_audio_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    total_available = min(len(goals), len(prompts))
    if args.max_samples is None or args.max_samples <= 0:
        total = total_available
    else:
        total = min(total_available, args.max_samples)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[\n")

    print(f"[INFO] Total available samples: {total_available}", flush=True)
    print(f"[INFO] Selected samples: {total}", flush=True)
    print(f"[INFO] Streaming results to: {output_file}", flush=True)
    print(f"[INFO] Using API URL: {api_url}", flush=True)
    
    results = []

    run_start = time.time()

    for i in range(total):
        item_start = time.time()
        goal_item = goals[i]
        prompt_item = prompts[i]

        goal = goal_item.get("goal")
        jailbreak_prompt = prompt_item.get("text_input")
        audio_file = prompt_item.get("audio_input")

        try:
            response = call_vllm(
                audio_path=audio_file,
                prompt=jailbreak_prompt,
                api_url=api_url,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"[ERROR] index {i} call_vllm failed: {e}", flush=True)
            response = f"[ERROR_CALL_VLLM] {e}"

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

        elapsed = time.time() - run_start
        avg = elapsed / (i + 1)
        remaining = avg * (total - i - 1)
        one_cost = time.time() - item_start
        print(
            f"[PROGRESS] {model_name} {i + 1}/{total} "
            f"step={one_cost:.1f}s elapsed={format_seconds(elapsed)} "
            f"eta={format_seconds(remaining)}",
            flush=True,
        )

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
