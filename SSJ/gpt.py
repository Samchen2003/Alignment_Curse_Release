import argparse
import base64
import json
import time
from pathlib import Path
from openai import OpenAI
from strong_reject.evaluate import evaluate as sr_evaluate
from tqdm import tqdm
from prefix import test_prefixes

def _extract_text_from_response(response) -> str:
    message = response.choices[0].message

    audio_obj = getattr(message, "audio", None)
    if audio_obj is not None:
        transcript = getattr(audio_obj, "transcript", None)
        if transcript:
            return transcript

    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text_val = item.get("text") or item.get("output_text")
                if text_val:
                    parts.append(text_val)
            else:
                text_val = getattr(item, "text", None) or getattr(item, "output_text", None)
                if text_val:
                    parts.append(text_val)
        if parts:
            return "\n".join(parts)

    raise ValueError("empty response content")


def call_gpt4o_audio(
    client: OpenAI,
    model: str,
    audio_path: str,
    prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    max_retry: int = 5,
    retry_sleep: int = 1,
) -> str:
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    current_max_tokens = max_tokens
    force_text_only = False

    for attempt in range(1, max_retry + 1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                "temperature": temperature,
                "max_tokens": current_max_tokens,
            }

            if not force_text_only:
                kwargs["modalities"] = ["text", "audio"]
                kwargs["audio"] = {"voice": "alloy", "format": "wav"}

            response = client.chat.completions.create(**kwargs)
            return _extract_text_from_response(response)

        except Exception as e:
            err_msg = str(e).lower()
            hit_output_limit = "max_tokens or model output limit was reached" in err_msg

            if hit_output_limit and current_max_tokens < 8000:
                next_max_tokens = min(current_max_tokens * 2, 8000)
                print(
                    f"[WARN] GPT-4o try {attempt}/{max_retry} hit output limit; "
                    f"raise max_tokens {current_max_tokens} -> {next_max_tokens}",
                    flush=True,
                )
                current_max_tokens = next_max_tokens
            elif hit_output_limit and not force_text_only:
                force_text_only = True
                print(
                    f"[WARN] GPT-4o try {attempt}/{max_retry} still hit output limit; "
                    "switching to text-only output mode",
                    flush=True,
                )
            else:
                print(f"[WARN] GPT-4o try {attempt}/{max_retry} failed: {e}", flush=True)

            if attempt < max_retry:
                time.sleep(retry_sleep)

    raise RuntimeError("GPT-4o retries exhausted")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-4o audio eval over jailbreak prompts.")
    parser.add_argument("--goal-target-path", typer=Path, required=True)
    parser.add_argument("--prompt-audio-path", typer=Path, required=True)
    parser.add_argument("--output-dir", typer=str, default="result")
    parser.add_argument("--output-file", default="gpt.json")
    parser.add_argument("--model", default="gpt-4o-audio-preview-2025-06-03")
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retry", type=int, default=5)
    parser.add_argument("--retry-sleep", type=int, default=1)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    goal_target_path = Path(args.goal_target_path)
    prompt_audio_path = Path(args.prompt_audio_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_file

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
    print(f"[INFO] Streaming results to: {output_file}", flush=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[\n")

    client = OpenAI()
    results = []
    run_start = time.time()

    for i in tqdm(range(total), desc="Running GPT-4o", ncols=80):
        item_start = time.time()
        goal_item = goals[i]
        prompt_item = prompts[i]

        goal = goal_item.get("goal")
        jailbreak_prompt = prompt_item.get("text_input")
        audio_file = prompt_item.get("audio_input")

        try:
            response = call_gpt4o_audio(
                client=client,
                model=args.model,
                audio_path=audio_file,
                prompt=jailbreak_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_retry=args.max_retry,
                retry_sleep=args.retry_sleep,
            )
        except Exception as e:
            print(f"[ERROR] index {i} GPT-4o failed: {e}", flush=True)
            response = f"[ERROR_GPT4O] {e}"

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
            f"[PROGRESS] gpt4o {i + 1}/{total} "
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
