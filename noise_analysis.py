"""Evaluate ASR robustness with noise and spectral subtraction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import jiwer
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import pipeline

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class SpeechSample:
    audio: np.ndarray
    reference: str
    name: str


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))) + 1e-12)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if source_rate != target_rate:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=target_rate)
    return audio


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix normalized noise so that speech/noise power ratio equals snr_db."""
    speech = np.asarray(speech, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if noise.size == 0:
        raise ValueError("Noise segment is empty")

    if noise.size < speech.size:
        repeats = int(np.ceil(speech.size / noise.size))
        noise = np.tile(noise, repeats)[: speech.size]
    else:
        start = np.random.randint(0, noise.size - speech.size + 1) if noise.size > speech.size else 0
        noise = noise[start : start + speech.size]

    scale = rms(speech) / (rms(noise) * (10.0 ** (snr_db / 20.0)))
    mixed = speech + scale * noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed /= peak
    return mixed.astype(np.float32)


def spectral_subtract(
    noisy: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    noise_duration: float = 0.25,
    alpha: float = 2.0,
    beta: float = 0.02,
) -> np.ndarray:
    """Simple FFT spectral subtraction using the beginning as a noise estimate."""
    noisy = np.asarray(noisy, dtype=np.float32)
    n_fft = 512
    hop = 160
    win = np.hanning(n_fft)

    noise_samples = max(n_fft, int(noise_duration * sample_rate))
    noise_samples = min(noise_samples, noisy.size)
    noise_frames = librosa.stft(
        noisy[:noise_samples],
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
    )
    noise_profile = np.mean(np.abs(noise_frames), axis=1)

    spectrum = librosa.stft(noisy, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win)
    magnitude, phase = np.abs(spectrum), np.angle(spectrum)
    cleaned_magnitude = np.maximum(magnitude - alpha * noise_profile[:, None], beta * magnitude)
    cleaned = librosa.istft(
        cleaned_magnitude * np.exp(1j * phase),
        hop_length=hop,
        win_length=n_fft,
        window=win,
        length=noisy.size,
    )
    peak = float(np.max(np.abs(cleaned))) if cleaned.size else 0.0
    if peak > 1.0:
        cleaned /= peak
    return cleaned.astype(np.float32)


def iter_local_audio(path: Path) -> Iterator[Path]:
    for pattern in ("*.flac", "*.wav", "*.mp3", "*.ogg"):
        yield from sorted(path.rglob(pattern))


def load_local_librispeech(root: Path, limit: int | None) -> list[SpeechSample]:
    transcriptions: dict[str, str] = {}
    for file in root.rglob("*"):
        if file.suffix.lower() == ".txt":
            for line in file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    transcriptions[parts[0]] = parts[1]

    samples: list[SpeechSample] = []
    for audio_path in iter_local_audio(root):
        key = audio_path.stem
        if key not in transcriptions:
            continue
        audio, rate = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        samples.append(SpeechSample(resample_audio(audio, rate), transcriptions[key], key))
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ValueError(
            f"No LibriSpeech transcripts/audio pairs found under {root}. "
            "Use the directory containing flac files and transcription .txt files."
        )
    return samples


def load_huggingface_librispeech(limit: int | None) -> list[SpeechSample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to use Hugging Face LibriSpeech.") from exc

    dataset = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
    samples: list[SpeechSample] = []
    for index, item in enumerate(dataset):
        audio = item["audio"]
        samples.append(
            SpeechSample(
                resample_audio(np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])),
                item["text"],
                f"librispeech_{index}",
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def load_local_noise(root: Path) -> list[np.ndarray]:
    noise: list[np.ndarray] = []
    for path in iter_local_audio(root):
        audio, rate = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        noise.append(resample_audio(audio, rate))
    if not noise:
        raise ValueError(f"No audio noise files found under {root}")
    return noise


def load_huggingface_noise(dataset_name: str, limit: int = 100) -> list[np.ndarray]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to use Hugging Face noise.") from exc

    dataset = load_dataset(dataset_name, split="test", streaming=True)
    noise: list[np.ndarray] = []
    for index, item in enumerate(dataset):
        audio = item["audio"]
        noise.append(
            resample_audio(
                np.asarray(audio["array"], dtype=np.float32),
                int(audio["sampling_rate"]),
            )
        )
        if len(noise) >= limit:
            break
    if not noise:
        raise ValueError(f"No audio noise files found in {dataset_name}")
    return noise


def build_asr(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )


def transcribe(asr, audio: np.ndarray) -> str:
    result = asr(
        {"array": np.asarray(audio, dtype=np.float32), "sampling_rate": SAMPLE_RATE},
    )
    return str(result["text"]).strip()


def evaluate(
    samples: Iterable[SpeechSample],
    noise_files: list[np.ndarray],
    snrs: Iterable[float],
    asr,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    rows: list[dict[str, object]] = []

    for sample in tqdm(list(samples), desc="Samples"):
        audio = sample.audio
        if audio.size < SAMPLE_RATE:
            continue

        clean_text = transcribe(asr, audio)
        rows.append(
            {
                "sample": sample.name,
                "pipeline": "clean",
                "snr_db": "none",
                "reference": sample.reference,
                "hypothesis": clean_text,
                "wer": jiwer.wer(sample.reference, clean_text),
                "cer": jiwer.cer(sample.reference, clean_text),
            }
        )

        noise_index = int(rng.integers(0, len(noise_files)))
        noise = noise_files[noise_index]

        for snr in snrs:
            noisy = mix_at_snr(audio, noise, float(snr))
            enhanced = spectral_subtract(noisy)

            noisy_text = transcribe(asr, noisy)
            enhanced_text = transcribe(asr, enhanced)

            for pipeline_name, hypothesis in (
                ("noisy", noisy_text),
                ("spectral_subtraction", enhanced_text),
            ):
                current_snr = "none" if pipeline_name == "clean" else float(snr)
                rows.append(
                    {
                        "sample": sample.name,
                        "pipeline": pipeline_name,
                        "snr_db": current_snr,
                        "reference": sample.reference,
                        "hypothesis": hypothesis,
                        "wer": jiwer.wer(sample.reference, hypothesis),
                        "cer": jiwer.cer(sample.reference, hypothesis),
                    }
                )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech-root", type=Path, help="Local LibriSpeech root")
    parser.add_argument("--musan-root", type=Path, help="Local MUSAN root")
    parser.add_argument(
        "--musan-dataset",
        default="noisy-alpaca-test/MUSAN-noise-audio-only",
        help="Hugging Face noise dataset (used when --musan-root is omitted)",
    )
    parser.add_argument("--snrs", nargs="+", type=float, default=[0.0, 10.0, 20.0])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        default="facebook/wav2vec2-base-960h",
        help="Hugging Face ASR model",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    if args.librispeech_root:
        samples = load_local_librispeech(args.librispeech_root, args.limit)
    else:
        samples = load_huggingface_librispeech(args.limit)

    if args.musan_root:
        noise_files = load_local_noise(args.musan_root)
    else:
        noise_files = load_huggingface_noise(args.musan_dataset, max(args.limit * 2, 20))
    asr = build_asr(args.model)
    raw_results = evaluate(samples, noise_files, args.snrs, asr, args.seed)

    summary = (
        raw_results.groupby(["pipeline", "snr_db"], dropna=False)[["wer", "cer"]]
        .agg(["mean", "std", "count"])
        .round(4)
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_results.to_csv(args.output_dir / "asr_results.csv", index=False)
    summary.to_csv(args.output_dir / "asr_summary.csv", index=False)

    print("\nASR robustness summary")
    print(summary.to_string(index=False))
    print(f"\nSaved detailed results to {args.output_dir / 'asr_results.csv'}")
    print(f"Saved summary to {args.output_dir / 'asr_summary.csv'}")


if __name__ == "__main__":
    main()
