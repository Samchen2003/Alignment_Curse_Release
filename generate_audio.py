import json
import os
import argparse
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio from JSON prompts using OpenAI TTS.")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to input JSON file.")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Path to output JSON file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store generated audio.")
    parser.add_argument("--voice", type=str, default="alloy",
                        help="Voice to use for TTS. Default: alloy.")
    parser.add_argument("--audio_format", type=str, default="wav",
                        choices=["wav", "mp3"],
                        help="Audio format to save. Default: wav.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-tts-2025-03-20",
                        help="OpenAI model for TTS. Default: gpt-4o-mini-tts.")
    parser.add_argument("--text_key", type=str, default="jailbreak_prompt",
                        help="Key in JSON objects that contains the text. Default: 'prompt'.")

    return parser.parse_args()

def main():
    args = parse_args()
    client = OpenAI()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    results = []

    for idx, item in enumerate(data):
        text = item.get(args.text_key, "")

        if not text:
            print(f"Item {idx} missing key '{args.text_key}', skipping.")
            continue

        print(f"Generating audio for item {idx}...")

        output_path = os.path.join(
            args.output_dir,
            f"{idx:04d}.{args.audio_format}"
        )

        with client.audio.speech.with_streaming_response.create(
            model=args.model,
            voice=args.voice,
            input=text,
        ) as response:
            response.stream_to_file(output_path)

        print(f"Saved: {output_path}")
        
        try:
            entry = {
                "goal": item["goal"],
                "jailbreak_prompt": item["jailbreak_prompt"],
                "text_response": item["text_response"],
                "text_score": item["text_score"],
                "audio_file": output_path           
            }
            
        except:
            entry = {
                "goal": item["goal"],
                "jailbreak_prompt": item["jailbreak_prompt"],
                "audio_file": output_path           
            }
            
        results.append(entry)
    
    
    output_json = args.output_json
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    main()