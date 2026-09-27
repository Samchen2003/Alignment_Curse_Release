import json
import os
import argparse
from openai import OpenAI


LANGUAGES = [
    ("en", "English"),
    ("fr", "French"),
    ("de", "German"),
    ("it", "Italian"),
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multilingual audio from JSON prompts using OpenAI TTS."
    )
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
    parser.add_argument("--tts_model", type=str, default="gpt-4o-mini-tts-2025-03-20",
                        help="OpenAI TTS model. Default: gpt-4o-mini-tts-2025-03-20.")
    parser.add_argument("--translation_model", type=str, default="gpt-3.5-turbo",
                        help="OpenAI model used to translate text before TTS.")
    parser.add_argument("--text_key", type=str, default="goal",
                        help="Key in JSON objects that contains the text. Default: 'goal'.")
    return parser.parse_args()

def translate_text(client, model, text, target_language):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise translation engine. "
                    "Return only the translated text, with no quotes, notes, or explanations."
                ),
            },
            {
                "role": "user",
                "content": f"Translate the following text into {target_language}:\n\n{text}",
            },
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

def generate_tts(client, model, voice, text, output_path):
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
    ) as response:
        response.stream_to_file(output_path)

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

        print(f"Generating 4 audio files for item {idx}...")

        item_dir = os.path.join(args.output_dir, f"{idx:04d}")
        os.makedirs(item_dir, exist_ok=True)

        audio_files = {}
        spoken_texts = {}

        for lang_code, lang_name in LANGUAGES:
            if lang_code == "en":
                translated_text = text 
            else:
                translated_text = translate_text(
                    client=client,
                    model=args.translation_model,
                    text=text,
                    target_language=lang_name,
                )

            spoken_texts[lang_code] = translated_text
            print(translated_text)

            output_path = os.path.join(
                item_dir,
                f"{idx:04d}_{lang_code}.{args.audio_format}"
            )

            print(f"  Generating {lang_name} audio...")
            generate_tts(
                client=client,
                model=args.tts_model,
                voice=args.voice,
                text=translated_text,
                output_path=output_path,
            )

            audio_files[lang_code] = output_path
            print(f"  Saved: {output_path}")
            
        entry = dict(item)
        entry["audio_files"] = audio_files
        entry["spoken_texts"] = spoken_texts
        results.append(entry)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done!")

if __name__ == "__main__":
    main()