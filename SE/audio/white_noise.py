import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

DEFAULT_START_TIME = 0.0
DEFAULT_END_TIME = 1.0
DEFAULT_GAIN = 5.0
DEFAULT_NOISE_SCALE = 1.5
DEFAULT_CLIP_MIN, DEFAULT_CLIP_MAX = -1.0, 1.0

AUDIO_EXTS = {".wav"}

def process_audio(in_path, out_path, start_time, end_time, gain, noise_scale, clip_min, clip_max):
    y, sr = librosa.load(in_path, sr=None, mono=False)
    if y.ndim == 1:
        y = np.expand_dims(y, 0)
    n_samples = y.shape[1]
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    start_sample = max(0, start_sample)
    end_sample = min(n_samples, end_sample)

    if start_sample < end_sample:
        y[:, start_sample:end_sample] *= gain
        for ch in range(y.shape[0]):
            seg = y[ch, start_sample:end_sample]
            seg_std = np.std(seg)
            noise_std = seg_std * noise_scale if seg_std > 0 else 1e-4
            noise = np.random.normal(0.0, noise_std, size=seg.shape)
            y[ch, start_sample:end_sample] = seg + noise

    y = np.clip(y, clip_min, clip_max)
    sf.write(out_path, y.T, sr)

def parse_args():
    parser = argparse.ArgumentParser(description="Amplify a segment and add white noise to wav files.")
    parser.add_argument("--input-dir", required=True, help="Input directory containing wav files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed wav files.")
    parser.add_argument("--start-time", type=float, default=DEFAULT_START_TIME, help=f"Start time in seconds (default: {DEFAULT_START_TIME}).")
    parser.add_argument("--end-time", type=float, default=DEFAULT_END_TIME, help=f"End time in seconds (default: {DEFAULT_END_TIME}).")
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN, help=f"Gain multiplier for selected segment (default: {DEFAULT_GAIN}).")
    parser.add_argument("--noise-scale", type=float, default=DEFAULT_NOISE_SCALE, help=f"Noise std multiplier vs segment std (default: {DEFAULT_NOISE_SCALE}).")
    parser.add_argument("--clip-min", type=float, default=DEFAULT_CLIP_MIN, help=f"Minimum clip value (default: {DEFAULT_CLIP_MIN}).")
    parser.add_argument("--clip-max", type=float, default=DEFAULT_CLIP_MAX, help=f"Maximum clip value (default: {DEFAULT_CLIP_MAX}).")
    return parser.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[WARN] input dir not found: {in_dir}")
        return

    wav_files = sorted([f for f in in_dir.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS])
    print(f"[INFO] found {len(wav_files)} wav files in {in_dir}, output -> {out_dir}")

    for wav in wav_files:
        out_path = out_dir / wav.name
        if out_path.exists():
            print(f"[SKIP] {out_path.name} (already exists)")
            continue
        try:
            print(f"[PROCESS] {wav.name}")
            process_audio(
                str(wav),
                str(out_path),
                args.start_time,
                args.end_time,
                args.gain,
                args.noise_scale,
                args.clip_min,
                args.clip_max,
            )
            print(f"[OK] saved -> {out_path}")
        except Exception as e:
            print(f"[ERROR] {wav.name}: {e}")

    print("All done.")

if __name__ == "__main__":
    main()
