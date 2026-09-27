import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from random import uniform
from typing import Any, Dict, List, Tuple

import torch
from strong_reject.evaluate import evaluate as sr_evaluate
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from prefix import test_prefixes

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

DEFAULT_OUTPUT_DIR = Path("result")
DEFAULT_LIMIT = 100
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_DO_SAMPLE = False
DEFAULT_MAX_RETRY = 3
DEFAULT_BASE_BACKOFF = 1.0
DEFAULT_BACKOFF_CAP = 30.0
DEFAULT_DTYPE = "float16"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("omni_batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAC audio evaluation with local InteractiveOmni model.")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing .wav files.")
    parser.add_argument("--revised-json", type=Path, required=True, help="Revised queries JSON path.")
    parser.add_argument("--goals-json", type=Path, required=True, help="Goals JSON path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--output-path", type=Path, default=None, help="Full output JSON path.")
    parser.add_argument("--model-path", required=True, help="Path to local model.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=DEFAULT_DTYPE, help="Model load dtype.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Maximum number of items to process.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Generation max_new_tokens.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Generation temperature.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling for generation.")
    parser.add_argument("--max-retry", type=int, default=DEFAULT_MAX_RETRY, help="Retry attempts for model inference.")
    parser.add_argument("--base-backoff", type=float, default=DEFAULT_BASE_BACKOFF, help="Backoff base seconds.")
    parser.add_argument("--backoff-cap", type=float, default=DEFAULT_BACKOFF_CAP, help="Backoff cap seconds.")
    return parser.parse_args()


def resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


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
        logger.exception("flexible_extract_dialogue failed: %s", e)
        return json.dumps(item, ensure_ascii=False)


def exponential_backoff_sleep(attempt: int, base_backoff: float, backoff_cap: float) -> None:
    backoff = min(backoff_cap, base_backoff * (2 ** (attempt - 1)))
    jitter = uniform(0, backoff * 0.1)
    time.sleep(backoff + jitter)


def load_model_and_tokenizer(
    model_path: str,
    dtype_str: str,
) -> Tuple[Any, Any]:
    dtype = resolve_dtype(dtype_str)

    logger.info("Loading tokenizer from %s ...", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    logger.info("Loading model from %s with device_map=auto + offload ...", model_path)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    logger.info("Model loaded. Device map: %s", getattr(model, "hf_device_map", None))
    return tokenizer, model


def get_model_response(
    audio_path: str,
    text_input: str,
    tokenizer: Any,
    model: Any,
    generation_config: Dict[str, Any],
    max_retry: int,
    base_backoff: float,
    backoff_cap: float,
) -> str:
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": text_input},
            ],
        }
    ]

    last_exc = None
    engine_fallback_applied = False
    for attempt in range(1, max_retry + 1):
        try:
            with torch.no_grad():
                try:
                    response = model.chat(tokenizer, messages, generation_config=generation_config)
                except TypeError:
                    response = model.chat(tokenizer, generation_config, messages)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            last_exc = e
            err_text = str(e).lower()
            if "unable to find an engine to execute this computation" in err_text and not engine_fallback_applied:
                torch.backends.cudnn.enabled = False
                engine_fallback_applied = True
                logger.warning("Inference engine error detected; disabled cuDNN and retrying for %s", audio_path)
                continue
            logger.warning("Model inference attempt %d failed for %s: %s", attempt, audio_path, e)
            if attempt < max_retry:
                exponential_backoff_sleep(attempt, base_backoff=base_backoff, backoff_cap=backoff_cap)
            else:
                raise
    raise RuntimeError(f"get_model_response failed: {last_exc}")


def strongreject_score(prompt: str, response: str) -> float:
    try:
        res = sr_evaluate(prompt, response, ["strongreject_rubric"])
        if isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], dict) and "score" in res[0]:
            return float(res[0]["score"])
        return 0.0
    except Exception as e:
        logger.exception("strong_reject failed: %s", e)
        return 0.0


def main() -> None:
    args = parse_args()
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
    }

    if not args.goals_json.exists():
        logger.error("Goals JSON not found at: %s", args.goals_json)
        return
    if not args.revised_json.exists():
        logger.error("Revised queries JSON not found at: %s", args.revised_json)
        return

    goals_list = load_as_list(args.goals_json)


    revised_list = load_as_list(args.revised_json)
    wav_files = list_wav_files(args.audio_dir, limit=args.limit)
    if len(wav_files) == 0:
        logger.error("No .wav files found in %s", args.audio_dir)
        return

    counts = [len(wav_files), len(revised_list)]
    if goals_list:
        counts.append(len(goals_list))
    count = min(counts)
    if count < max(counts):
        logger.warning("Mismatch counts - will process %d items (min of lists).", count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sanitized_model_name = Path(args.model_path).name.replace("/", "_")
    output_path = args.output_path or (args.output_dir / f"{sanitized_model_name}_response.json")

    logger.info(
        "Found %d wav files, %d revised entries, %d dialogues (fallback), %d goals",
        len(wav_files),
        len(revised_list),
        len(goals_list),
    )
    logger.info("Will process %d items. Output: %s", count, output_path)

    tokenizer, model = load_model_and_tokenizer(
        model_path=args.model_path,
        dtype_str=args.dtype,
    )

    fout = open(output_path, "w", encoding="utf-8")
    fout.write("[\n")
    fout.flush()
    os.fsync(fout.fileno())
    first_entry = True
    results=[]
    try:
        for i in tqdm(range(count), desc="Processing", ncols=80):
            audio_path = wav_files[i]
            revised_item = revised_list[i]
            goal_item = goals_list[i] if i < len(goals_list) else {}

            extracted = flexible_extract_dialogue(revised_item)
            revised_query_text=revised_item["revised_query"].strip()


            jailbreak_prompt = (extracted + "\n\n" + revised_query_text).strip()
            try:
                response_text = get_model_response(
                    str(audio_path),
                    revised_query_text,
                    tokenizer=tokenizer,
                    model=model,
                    generation_config=generation_config,
                    max_retry=args.max_retry,
                    base_backoff=args.base_backoff,
                    backoff_cap=args.backoff_cap,
                )
            except Exception as e:
                logger.exception("Index %d model inference failed", i)
                response_text = f"[ERROR_OMNI_LOCAL] {e}"

            score = strongreject_score(jailbreak_prompt, response_text) if response_text else 0.0
            out_obj = {
                "goal": goal_item.get("goal") if isinstance(goal_item, dict) and "goal" in goal_item else None,
                "target": goal_item.get("target") if isinstance(goal_item, dict) and "target" in goal_item else None,
                "jailbreak_prompt": jailbreak_prompt,
                "audio_file": str(audio_path),
                "audio_response": response_text,
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
