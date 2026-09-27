import argparse
import base64
import json
import logging
import os
import time
from pathlib import Path
from random import uniform
from typing import Any, Dict, List

from openai import OpenAI
from strong_reject.evaluate import evaluate as sr_evaluate
from tqdm import tqdm
from prefix import test_prefixes

DEFAULT_OUTPUT_DIR = Path("result")
DEFAULT_MODEL_NAME = "gpt-4o-audio-preview-2025-06-03"
DEFAULT_LIMIT = 100
DEFAULT_MAX_RETRY = 5
DEFAULT_BASE_BACKOFF = 1.0
DEFAULT_BACKOFF_CAP = 60.0
DEFAULT_MAX_TOKENS = 512

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAC audio evaluation with GPT audio chat completions.")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing .wav files.")
    parser.add_argument("--revised-json", type=Path, required=True, help="Revised queries JSON path.")
    parser.add_argument("--goals-json", type=Path, required=True, help="Goals JSON path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--output-path", type=Path, default=None, help="Full output JSON path.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="OpenAI model name.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Maximum number of items to process.")
    parser.add_argument("--max-retry", type=int, default=DEFAULT_MAX_RETRY, help="Retry attempts for model calls.")
    parser.add_argument("--base-backoff", type=float, default=DEFAULT_BASE_BACKOFF, help="Backoff base seconds.")
    parser.add_argument("--backoff-cap", type=float, default=DEFAULT_BACKOFF_CAP, help="Backoff cap seconds.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="max_tokens for chat completion.")
    return parser.parse_args()


def get_client() -> OpenAI:
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        raise RuntimeError("OPENAI_API_KEY must be set in environment (do NOT hardcode keys).")
    return OpenAI()


def load_as_list(json_path: Path) -> List[Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        return list(data.values())
    if isinstance(data, list):
        return data
    return [data]


def list_wav_files(audio_dir: Path, limit: int) -> List[Path]:
    files = sorted([p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"])
    return files[: min(limit, len(files))]


def flexible_extract_dialogue(item: Any) -> str:
    try:
        if item is None:
            return ""
        if isinstance(item, str):
            return item.strip()
        candidate_keys = ["dialogue_script", "dialogue", "prompt", "conversation", "messages", "turns"]
        if isinstance(item, dict):
            for k in candidate_keys:
                if k in item:
                    val = item[k]
                    if isinstance(val, str):
                        return val.strip()
                    if isinstance(val, list):
                        parts = []
                        for el in val:
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
                    if isinstance(val, dict):
                        return flexible_extract_dialogue(val)
            for f in ("text", "content", "utterance", "message"):
                if f in item and isinstance(item[f], str):
                    return item[f].strip()
            parts = [f"{k}: {v.strip()}" for k, v in item.items() if isinstance(v, str)]
            if parts:
                return "\n".join(parts).strip()
        if isinstance(item, list):
            collected = [flexible_extract_dialogue(el) for el in item]
            return "\n".join([c for c in collected if c])
        return json.dumps(item, ensure_ascii=False)
    except Exception as e:
        logger.exception("Failed to extract dialogue: %s", e)
        return json.dumps(item, ensure_ascii=False)


def exponential_backoff_sleep(attempt: int, base_backoff: float, backoff_cap: float) -> None:
    backoff = min(backoff_cap, base_backoff * (2 ** (attempt - 1)))
    jitter = uniform(0, backoff * 0.1)
    time.sleep(backoff + jitter)


def call_gpt4o_audio(
    audio_path: Path,
    text_input: str,
    client: OpenAI,
    model: str,
    max_retry: int,
    base_backoff: float,
    backoff_cap: float,
    max_tokens: int,
) -> Dict[str, Any]:
    if not audio_path.exists():
        return {"ok": False, "transcript": "", "attempts": 0, "latency": None, "error": f"audio file not found: {audio_path}"}

    try:
        with open(audio_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return {"ok": False, "transcript": "", "attempts": 0, "latency": None, "error": f"failed to read audio: {e}"}

    last_exc = None
    start_ts = time.time()
    for attempt in range(1, max_retry + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                modalities=["text", "audio"],
                audio={"voice": "alloy", "format": "wav"},
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                            {"type": "text", "text": text_input},
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )

            latency = time.time() - start_ts
            transcript = ""
            try:
                transcript = response.choices[0].message.audio.transcript
            except Exception:
                transcript = response.choices[0].message.content if hasattr(response.choices[0].message, "content") else str(response)
            return {"ok": True, "transcript": transcript, "attempts": attempt, "latency": latency, "error": None}
        except Exception as e:
            last_exc = e
            logger.warning("Model call attempt %d failed for %s: %s", attempt, audio_path.name, e)
            if attempt < max_retry:
                exponential_backoff_sleep(attempt, base_backoff=base_backoff, backoff_cap=backoff_cap)

    return {"ok": False, "transcript": "", "attempts": max_retry, "latency": None, "error": str(last_exc)}


def strongreject_score(prompt: str, response_text: str) -> float:
    try:
        res = sr_evaluate(prompt, response_text, ["strongreject_rubric"])
        if isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], dict) and "score" in res[0]:
            return float(res[0]["score"])
        return 0.0
    except Exception as e:
        logger.exception("strong_reject evaluation failed: %s", e)
        return 0.0


def main() -> None:
    args = parse_args()
    logger.info("Starting incremental run: audio + revised_query sent; jailbreak_prompt recorded only.")

    if not args.goals_json.exists():
        logger.error("Goals JSON not found at: %s", args.goals_json)
        return
    if not args.revised_json.exists():
        logger.error("Revised JSON not found at: %s", args.revised_json)
        return

    try:
        goals_list = load_as_list(args.goals_json)
        revised_list = load_as_list(args.revised_json)
    except Exception as e:
        logger.exception("Failed to load one or more JSON inputs: %s", e)
        return

    wav_files = list_wav_files(args.audio_dir, limit=args.limit)
    if len(wav_files) == 0:
        logger.error("No .wav files found in %s", args.audio_dir)
        return
    logger.info("Found %d wav files.", len(wav_files))

    count = min(len(wav_files), len(revised_list), len(goals_list))
    if count < max(len(wav_files), len(revised_list), len(goals_list)):
        logger.warning("Mismatch counts - will process %d items.", count)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_path or (output_dir / f"{args.model_name.replace('/', '_')}_response.json")

    client = get_client()

    fout = None
    first_entry = True
    results=[]
    try:
        fout = open(output_path, "w", encoding="utf-8")
        fout.write("[\n")
        fout.flush()
        os.fsync(fout.fileno())

        for i in tqdm(range(count), desc="Processing", ncols=80):
            audio_path = wav_files[i]
            goal_item = goals_list[i]
            
            revised_query_text=revised_list[i]["revised_query"].strip()
            extracted = flexible_extract_dialogue(revised_list[i])

            jailbreak_prompt = (extracted + "\n\n" + revised_query_text).strip()

            gpt_res = call_gpt4o_audio(
                audio_path,
                revised_query_text,
                client=client,
                model=args.model_name,
                max_retry=args.max_retry,
                base_backoff=args.base_backoff,
                backoff_cap=args.backoff_cap,
                max_tokens=args.max_tokens,
            )
            transcript = gpt_res.get("transcript", "") if gpt_res.get("ok") else ""
            score = strongreject_score(jailbreak_prompt, transcript) if transcript else 0.0

            out_obj = {
                "goal": goal_item.get("goal") if isinstance(goal_item, dict) and "goal" in goal_item else None,
                "target": goal_item.get("target") if isinstance(goal_item, dict) and "target" in goal_item else None,
                "jailbreak_prompt": jailbreak_prompt,
                "audio_file": str(audio_path),
                "audio_response": transcript,
                "audio_score": score,
            }
            results.append(out_obj)

            entry_json_pretty = json.dumps(out_obj, ensure_ascii=False, indent=2)
            if first_entry:
                fout.write(entry_json_pretty)
                first_entry = False
            else:
                fout.write(",\n" + entry_json_pretty)
            fout.flush()
            os.fsync(fout.fileno())

        fout.write("\n]\n")
        fout.flush()
        os.fsync(fout.fileno())
        logger.info("Finished. Wrote %d entries to %s", count, output_path)
    except Exception as e:
        logger.exception("Error during processing: %s", e)
        try:
            if fout and not fout.closed:
                fout.write("\n]\n")
                fout.flush()
                os.fsync(fout.fileno())
        except Exception:
            pass
        raise
    finally:
        if fout and not fout.closed:
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
