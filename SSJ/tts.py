from __future__ import annotations

import argparse
import json
import os
import tempfile
import wave
from pathlib import Path

from openai import OpenAI


def load_selected_words(masked_json_path: Path) -> list[str]:
    with masked_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Invalid or empty JSON list: {masked_json_path}")

    words: list[str] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"[WARN] item {i} is not an object, skip")
            words.append("")
            continue
        words.append(str(item.get("selected_word", "")).strip())
    return words


def synth_char_audio(client: OpenAI, ch: str, out_path: Path, voice: str) -> None:
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=ch,
        instructions="Pronounce exactly one character, clearly and briefly.",
        response_format="wav",
    ) as response:
        response.stream_to_file(out_path)

    with out_path.open("rb") as f:
        header = f.read(12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError(
            f"Generated file is not WAV for character {ch!r}: {out_path}"
        )


def merge_wavs(input_paths: list[Path], output_path: Path, gap_seconds: float = 0.15) -> None:
    if not input_paths:
        raise ValueError("No input wav files to merge")

    with wave.open(str(input_paths[0]), "rb") as first_wav:
        nchannels = first_wav.getnchannels()
        sampwidth = first_wav.getsampwidth()
        framerate = first_wav.getframerate()
        comptype = first_wav.getcomptype()
        compname = first_wav.getcompname()

    with wave.open(str(output_path), "wb") as out_wav:
        out_wav.setnchannels(nchannels)
        out_wav.setsampwidth(sampwidth)
        out_wav.setframerate(framerate)
        out_wav.setcomptype(comptype, compname)

        bytes_per_frame = nchannels * sampwidth
        silence_frames = int(max(gap_seconds, 0.0) * framerate)
        silence_chunk = b"\x00" * (silence_frames * bytes_per_frame)

        for idx, path in enumerate(input_paths):
            with wave.open(str(path), "rb") as in_wav:
                if (
                    in_wav.getnchannels() != nchannels
                    or in_wav.getsampwidth() != sampwidth
                    or in_wav.getframerate() != framerate
                    or in_wav.getcomptype() != comptype
                ):
                    raise ValueError(
                        f"Incompatible wav format in {path.name}; cannot merge safely"
                    )
                out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
                if idx != len(input_paths) - 1 and silence_frames > 0:
                    out_wav.writeframes(silence_chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--masked-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    masked_json = args.masked_json

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    selected_words = load_selected_words(masked_json)
    if not selected_words:
        raise ValueError("No selected_word entries found")


    with open(masked_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    voice = "coral"

    client = OpenAI()
    total = len(selected_words)
    print(f"Total items: {total}")
    results=[]

    for idx, selected_word in enumerate(selected_words):
        output_path = base_dir / f"item_{idx}.wav"
        if output_path.exists():
            print(f"[{idx + 1}/{total}] exists, skip -> {output_path.name}")
            continue

        chars = [c for c in selected_word if not c.isspace()]
        if not chars:
            print(f"[{idx + 1}/{total}] empty selected_word, skip")
            continue

        spelled_token = "-".join(chars)
        print(f"[{idx + 1}/{total}] word={selected_word} spelling={spelled_token}")

        with tempfile.TemporaryDirectory(prefix=f"tts_chars_{idx}_", dir=str(base_dir)) as tmp_dir:
            segment_dir = Path(tmp_dir)
            segment_paths: list[Path] = []

            for ch_i, ch in enumerate(chars):
                seg_path = segment_dir / f"{ch_i:02d}_{ch}.wav"
                print(
                    f"[{idx + 1}/{total}] segment {ch_i + 1}/{len(chars)}: {ch} -> {seg_path.name}"
                )
                synth_char_audio(client, ch, seg_path, voice=voice)
                segment_paths.append(seg_path)

            merge_wavs(segment_paths, output_path, gap_seconds=0.18)
            print(f"[{idx + 1}/{total}] done -> {output_path.name}")
            
        one_result = {
            "id": data[idx]["id"],
            "audio_input": str(output_path),
            "text_input": data[idx]["text_prompt"]
        }
        
        results.append(one_result)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)



    print("All done.")


if __name__ == "__main__":
    main()
