import numpy as np

from src.noise_analysis import mix_at_snr, rms, spectral_subtract


def test_mix_at_snr_has_expected_level():
    rng = np.random.default_rng(7)
    speech = rng.normal(0, 0.1, 16_000)
    noise = rng.normal(0, 0.5, 16_000)

    mixed = mix_at_snr(speech, noise, 10.0)

    speech_rms = rms(speech)
    residual = mixed - speech
    expected_ratio = 10 ** (-10.0 / 20.0)
    assert abs(rms(residual) / speech_rms - expected_ratio) < 0.08


def test_spectral_subtraction_preserves_length():
    rng = np.random.default_rng(11)
    noisy = rng.normal(0, 0.1, 16_000).astype(np.float32)
    cleaned = spectral_subtract(noisy)
    assert cleaned.shape == noisy.shape
    assert np.isfinite(cleaned).all()
