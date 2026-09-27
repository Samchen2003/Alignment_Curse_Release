import argparse
import os
from pathlib import Path
from pydub import AudioSegment


NOISE_AUDIO_PATH = "bird.wav"

DEFAULT_TARGET_NOISE_DBFS = -30.0

AUDIO_EXTS = {".wav"}

DEFAULT_OVERWRITE = False

def load_and_normalize_noise(noise_path: str, target_dbfs: float) -> AudioSegment:
    noise = AudioSegment.from_file(noise_path)
    current_dbfs = noise.dBFS
    gain = target_dbfs - current_dbfs
    adjusted = noise.apply_gain(gain)
    print(f"[noise] loaded: '{noise_path}', current dBFS={current_dbfs:.2f}, target={target_dbfs:.2f}, applied gain={gain:.2f} dB")
    return adjusted

def match_noise_length(original: AudioSegment, noise: AudioSegment) -> AudioSegment:
    orig_len = len(original)  
    if len(noise) >= orig_len:
        return noise[:orig_len]
    else:
        repeats = orig_len // len(noise) + 1
        extended = noise * repeats
        return extended[:orig_len]

def process_dir(in_dir: str, out_dir: str, noise_seg: AudioSegment, overwrite: bool):
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[WARN] input dir not found: {in_path}")
        return

    wav_files = sorted([p for p in in_path.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS])
    print(f"[INFO] found {len(wav_files)} wav files in {in_path}, output -> {out_path}")

    for f in wav_files:
        try:
            src = AudioSegment.from_file(str(f))
        except Exception as e:
            print(f"[ERROR] failed to open '{f}': {e}")
            continue

        out_file = out_path / f.name
        if out_file.exists() and not overwrite:
            print(f"[SKIP] exists: {out_file.name}")
            continue

        noise_to_add = match_noise_length(src, noise_seg)

        try:
            mixed = src.overlay(noise_to_add)
            mixed.export(str(out_file), format="wav")
            print(f"[OK] {f.name} -> {out_file}")
        except Exception as e:
            print(f"[ERROR] mixing/export failed for '{f.name}': {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Overlay normalized noise on wav files in a directory.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing wav files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for mixed wav files.",
    )
    parser.add_argument(
        "--noise-audio-path",
        default=NOISE_AUDIO_PATH,
        help=f"Path to noise audio file (default: {NOISE_AUDIO_PATH}).",
    )
    parser.add_argument(
        "--target-noise-dbfs",
        type=float,
        default=DEFAULT_TARGET_NOISE_DBFS,
        help=f"Target noise loudness in dBFS (default: {DEFAULT_TARGET_NOISE_DBFS}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=DEFAULT_OVERWRITE,
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.noise_audio_path):
        print(f"[FATAL] noise file not found: {args.noise_audio_path}")
        return

    noise_seg = load_and_normalize_noise(args.noise_audio_path, args.target_noise_dbfs)

    process_dir(args.input_dir, args.output_dir, noise_seg, args.overwrite)

    print("All done.")

if __name__ == "__main__":
    main()
