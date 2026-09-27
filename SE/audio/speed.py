import argparse
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np


def _read_wav_pcm(path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 1:
        dtype = np.uint8
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        dtype = np.int16
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        dtype = np.int32
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    data = data.reshape(-1, channels)
    return data, sr


def _write_wav_pcm16(path, data, sr):
    clipped = np.clip(data, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm16.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

DEFAULT_SPEED_RATES = [0.5, 1.5]

def process_dir(input_dir, output_dir, speed_rates):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"[WARN] Input dir not found: {input_dir}")
        return

    audio_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() == ".wav"])
    print(f"[INFO] found {len(audio_files)} wav files in {input_dir}")

    sox_bin = shutil.which("sox")
    ffmpeg_bin = shutil.which("ffmpeg")

    if sox_bin:
        backend = "sox"
    elif ffmpeg_bin:
        backend = "ffmpeg"
        print("[WARN] 'sox' not found, fallback to 'ffmpeg'.")
    else:
        backend = "python"
        print("[WARN] Neither 'sox' nor 'ffmpeg' found, fallback to pure Python resampling.")

    for rate in speed_rates:
        if len(speed_rates) == 1:
            rate_output_dir = output_dir
        else:
            rate_output_dir = output_dir / f"rate_{rate}"

        rate_output_dir.mkdir(parents=True, exist_ok=True)

        for audio_file in audio_files:
            out_name = audio_file.name
            out_path = rate_output_dir / out_name

            if out_path.exists():
                print(f"[SKIP] {out_path}")
                continue

            if backend == "sox":
                cmd = [
                    sox_bin,
                    str(audio_file),
                    str(out_path),
                    "tempo",
                    "-s",
                    str(rate),
                ]
            else:
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(audio_file),
                    "-filter:a",
                    f"atempo={rate}",
                    str(out_path),
                ]

            try:
                if backend == "python":
                    wav, sr = _read_wav_pcm(audio_file)
                    in_len = wav.shape[0]
                    out_len = max(1, int(round(in_len / rate)))

                    old_idx = np.linspace(0, in_len - 1, num=in_len)
                    new_idx = np.linspace(0, in_len - 1, num=out_len)

                    stretched = np.empty((out_len, wav.shape[1]), dtype=np.float32)
                    for ch in range(wav.shape[1]):
                        stretched[:, ch] = np.interp(new_idx, old_idx, wav[:, ch])

                    _write_wav_pcm16(out_path, stretched, sr)
                else:
                    subprocess.run(cmd, check=True)
                print(f"[OK] x{rate}: {audio_file.name}")
            except FileNotFoundError as e:
                print(f"[ERROR] Command not found while processing {audio_file.name}: {e}")
                return
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] x{rate}: {audio_file.name} failed: {e}")


def _parse_speed_rates(value: str):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Speed rates must contain at least one numeric value.")

    rates = []
    for p in parts:
        try:
            rate = float(p)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid speed rate '{p}'.") from e
        if rate <= 0:
            raise argparse.ArgumentTypeError("Speed rates must be > 0.")
        rates.append(rate)
    return rates


def parse_args():
    parser = argparse.ArgumentParser(description="Apply speed perturbation to wav files in configured directories.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing wav files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for generated wav files. If multiple speed rates are set, per-rate subfolders are created here.",
    )
    parser.add_argument(
        "--speed-rates",
        type=_parse_speed_rates,
        default=DEFAULT_SPEED_RATES,
        help=f"Comma-separated speed rates (default: {','.join(str(r) for r in DEFAULT_SPEED_RATES)}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_dir(args.input_dir, args.output_dir, args.speed_rates)

    print("Speed augmentation finished.")


if __name__ == "__main__":
    main()
