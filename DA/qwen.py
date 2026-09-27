import argparse
import base64
import json
import os
import time
from pathlib import Path
from random import uniform
from typing import Any, Dict, List
from prefix import test_prefixes
import requests
from strong_reject.evaluate import evaluate as sr_evaluate

DEFAULT_OUTPUT_DIR = Path("result")
DEFAULT_MODEL_NAME = "qwen2.5-omni-7B"
DEFAULT_LIMIT = 100
DEFAULT_MAX_RETRY = 5
DEFAULT_BASE_BACKOFF = 1.0
DEFAULT_BACKOFF_CAP = 60.0
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_TOKENS = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAC audio evaluation against a Qwen-compatible chat completions endpoint.")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing .wav files.")
    parser.add_argument("--revised-json", type=Path, required=True, help="Revised queries JSON path.")
    parser.add_argument("--goals-json", type=Path, required=True, help="Goals JSON path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write output JSON.")
    parser.add_argument("--output-path", type=Path, default=None, help="Full path to output JSON.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model key used for default endpoint lookup.")
    parser.add_argument("--api-url", required=True, help="Override the chat completions endpoint.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Maximum number of items to process.")
    parser.add_argument("--max-retry", type=int, default=DEFAULT_MAX_RETRY, help="Retry attempts for HTTP calls.")
    parser.add_argument("--base-backoff", type=float, default=DEFAULT_BASE_BACKOFF, help="Backoff base seconds.")
    parser.add_argument("--backoff-cap", type=float, default=DEFAULT_BACKOFF_CAP, help="Backoff cap seconds.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout (seconds) for each request.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="max_tokens for the request payload.")
    return parser.parse_args()


def load_as_list(json_path: Path) -> List[Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    return list(data.values()) if isinstance(data, dict) else [data]






def list_wav_files(audio_dir: Path, limit: int) -> List[Path]:
    files = sorted([p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"])
    return files[: min(limit, len(files))]


def flexible_extract_dialogue(item: Any) -> str:
    try:
        if item is None:
            return ""
        if isinstance(item, str):
            return item.strip()
        keys = ["dialogue_script", "dialogue", "prompt", "conversation", "messages", "turns"]
        if isinstance(item, dict):
            for k in keys:
                if k in item:
                    v = item[k]
                    if isinstance(v, str):
                        return v.strip()
                    if isinstance(v, list):
                        parts = []
                        for el in v:
                            if isinstance(el, str):
                                parts.append(el.strip())
                            elif isinstance(el, dict):
                                text = el.get("utterance") or el.get("text") or el.get("content") or el.get("message")
                                speaker = el.get("speaker") or el.get("role") or el.get("from")
                                if isinstance(text, str):
                                    parts.append(f"{speaker}: {text.strip()}" if speaker else text.strip())
                                else:
                                    parts.append(json.dumps(el, ensure_ascii=False))
                            else:
                                parts.append(str(el))
                        return "\n".join(parts).strip()
                    if isinstance(v, dict):
                        return flexible_extract_dialogue(v)
            for f in ("text", "content", "utterance", "message"):
                if f in item and isinstance(item[f], str):
                    return item[f].strip()
            parts = [f"{k}: {v.strip()}" for k, v in item.items() if isinstance(v, str)]
            if parts:
                return "\n".join(parts).strip()
        if isinstance(item, list):
            out = [flexible_extract_dialogue(el) for el in item]
            return "\n".join([s for s in out if s])
        return json.dumps(item, ensure_ascii=False)
    except Exception:
        return json.dumps(item, ensure_ascii=False)


def exponential_backoff_sleep(attempt: int, base_backoff: float, backoff_cap: float) -> None:
    backoff = min(backoff_cap, base_backoff * (2 ** (attempt - 1)))
    jitter = uniform(0, backoff * 0.1)
    time.sleep(backoff + jitter)


def call_vllm(
    audio_path: str,
    text_input: str,
    api_url: str,
    timeout: int,
    max_retry: int,
    base_backoff: float,
    backoff_cap: float,
    max_tokens: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False, "transcript": "", "attempts": 0, "latency": None, "error": None}
    if not Path(audio_path).exists():
        result["error"] = f"audio file not found: {audio_path}"
        return result

    try:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        result["error"] = f"failed to read audio: {e}"
        return result

    payload = {
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                    {"type": "text", "text": text_input},
                ],
            },
        ],
    }

    last_exc = None
    start_ts = time.time()
    for attempt in range(1, max_retry + 1):
        result["attempts"] = attempt
        try:
            resp = requests.post(api_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            latency = time.time() - start_ts
            transcript = ""
            try:
                transcript = body["choices"][0]["message"].get("content", "")
                if not transcript and "choices" in body and body["choices"]:
                    transcript = str(body["choices"][0]["message"])
            except Exception:
                transcript = json.dumps(body, ensure_ascii=False)
            result.update({"ok": True, "transcript": transcript, "latency": latency, "error": None})
            return result
        except Exception as e:
            last_exc = e
            result["error"] = str(e)
            if attempt < max_retry:
                exponential_backoff_sleep(attempt, base_backoff=base_backoff, backoff_cap=backoff_cap)

    result["error"] = str(last_exc)
    return result


def strongreject_score(prompt: str, response: str) -> float:
    try:
        res = sr_evaluate(prompt, response, ["strongreject_rubric"])
        if isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], dict) and "score" in res[0]:
            return float(res[0]["score"])
        return 0.0
    except Exception:
        return 0.0


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    args = parse_args()

    api_url = args.api_url
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_path or (args.output_dir / f"{args.model_name.replace('/', '_')}_re_evaluate_100.json")


    goals_list: List[Any] = []
    if args.goals_json.exists():
        goals_list = load_as_list(args.goals_json)
    else:
        print(f"[WARN] Goals file not found at {args.goals_json}; goal/target will be null.")

    if not args.revised_json.exists():
        raise FileNotFoundError(f"Revised queries JSON not found at {args.revised_json}")
    revised_list = load_as_list(args.revised_json)


    wav_files = list_wav_files(args.audio_dir, limit=args.limit)
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav files found in {args.audio_dir}")

    lengths = [len(wav_files), len(revised_list)]
    if goals_list:
        lengths.append(len(goals_list))
    count = min(lengths)
    if args.limit is not None:
        count = min(count, args.limit)

    print(
        f"[INFO] Found {len(wav_files)} wavs, {len(revised_list)} revised entries, "
    )
    print(f"[INFO] Will process {count} items.")
    print(f"[INFO] API URL: {api_url}")
    print(f"[INFO] Output path: {output_path}")

    fout = open(output_path, "w", encoding="utf-8")
    fout.write("[\n")
    fout.flush()
    os.fsync(fout.fileno())
    first = True
    run_started_at = time.time()
    results=[]
    try:
        for i in range(count):
            item_started_at = time.time()
            audio_path = wav_files[i]
            revised_item = revised_list[i]
            goal_item = goals_list[i] if i < len(goals_list) else {}
            extracted = flexible_extract_dialogue(revised_item)
            revised_query_text=revised_item["revised_query"].strip()

            jailbreak_prompt = (extracted + "\n\n" + revised_query_text).strip()
            qwen_res = call_vllm(
                str(audio_path),
                revised_query_text,
                api_url=api_url,
                timeout=args.timeout,
                max_retry=args.max_retry,
                base_backoff=args.base_backoff,
                backoff_cap=args.backoff_cap,
                max_tokens=args.max_tokens,
            )
            transcript = qwen_res.get("transcript", "") if qwen_res.get("ok", False) else ""
            score = strongreject_score(jailbreak_prompt, transcript) if transcript else 0.0

            out_obj = {
                "goal": goal_item.get("goal") if isinstance(goal_item, dict) and "goal" in goal_item else None,
                "target": goal_item.get("target") if isinstance(goal_item, dict) and "target" in goal_item else None,
                "jailbreak_dialogue": extracted,
                "jailbreak_prompt": revised_query_text,
                "audio_file": str(audio_path),
                "audio_response": transcript,
                "audio_score": score,
            }
            results.append(out_obj)
   
            entry_pretty = json.dumps(out_obj, ensure_ascii=False, indent=2)
            if first:
                fout.write(entry_pretty)
                first = False
            else:
                fout.write(",\n" + entry_pretty)
            fout.flush()
            os.fsync(fout.fileno())

            completed = i + 1
            elapsed = time.time() - run_started_at
            avg_per_item = elapsed / completed
            remaining = avg_per_item * (count - completed)
            item_elapsed = time.time() - item_started_at
            print(
                f"[INFO] Progress {completed}/{count} | item {item_elapsed:.1f}s | "
                f"elapsed {format_duration(elapsed)} | ETA {format_duration(remaining)}"
            )

        fout.write("\n]\n")
        fout.flush()
        os.fsync(fout.fileno())
        print(f"[INFO] Done. Wrote {count} entries to {output_path}")
    except Exception as e:
        print("[ERROR] Exception during run:", e)
        try:
            if not fout.closed:
                fout.write("\n]\n")
                fout.flush()
                os.fsync(fout.fileno())
        except Exception:
            pass
        raise
    finally:
        if not fout.closed:
            fout.close()


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
    main()
