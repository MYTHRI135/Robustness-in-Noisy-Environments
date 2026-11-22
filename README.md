# Speech Recognition Robustness Under Noise

This project evaluates how café, traffic, music, and other environmental noise affect automatic speech recognition (ASR).

## Experiment

For clean LibriSpeech speech, MUSAN noise, and each SNR in `0 dB`, `10 dB`, and `20 dB`:

1. Mix clean speech and noise at the target signal-to-noise ratio.
2. Transcribe the clean, noisy, and spectral-subtraction-enhanced audio.
3. Compute Word Error Rate (WER) and Character Error Rate (CER).
4. Save detailed and aggregate results as CSV files.

## Setup

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

## Download datasets

You can use the Hugging Face LibriSpeech loader and a local MUSAN folder:

```powershell
python src/noise_analysis.py --musan-root C:\datasets\MUSAN --limit 100
```

For local LibriSpeech, pass the directory containing both the `.flac` files and LibriSpeech `.txt` transcript files:

```powershell
python src/noise_analysis.py --librispeech-root C:\datasets\LibriSpeech --musan-root C:\datasets\MUSAN --snrs 0 10 20 --limit 100
```

MUSAN has a standard `noise/free_sound`, `noise/sound-bible`, and `noise/technical` layout. Any folder of WAV/FLAC/MP3/OGG audio may be supplied to `--musan-root`.

The default ASR model is `facebook/wav2vec2-base-960h`. Override it with `--model`:

```powershell
python src/noise_analysis.py --musan-root C:\datasets\MUSAN --model openai/whisper-small
```

## Test

```powershell
python -m pytest tests
```

## Results

The experiment writes:

- `results/asr_results.csv`: per-sample references, hypotheses, WER, and CER
- `results/asr_summary.csv`: mean, standard deviation, and count for each pipeline and SNR
