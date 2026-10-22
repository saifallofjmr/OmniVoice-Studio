import numpy as np
import pytest

from omnivoice.scripts import live_pitch
from omnivoice.scripts.live_pitch import (
    RegisterGuard,
    VoicedPitch,
    YinTracker,
    estimate_voiced_pitch,
    map_f0_to_target_register,
    semitones_between,
    shift_pitch_preserving_formants,
)


def harmonic_voice(f0_hz, seconds=2.0, sample_rate=16000, amplitude=0.3, vibrato=0.01):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    instantaneous = f0_hz * (1.0 + vibrato * np.sin(2.0 * np.pi * 4.5 * t))
    phase = 2.0 * np.pi * np.cumsum(instantaneous) / sample_rate
    signal = sum(np.sin(harmonic * phase) / harmonic for harmonic in range(1, 14))
    fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.05)
    return (amplitude * signal * fade).astype(np.float32)


def test_semitones_between_is_signed_and_octave_exact():
    assert semitones_between(110.0, 220.0) == pytest.approx(12.0)
    assert semitones_between(220.0, 110.0) == pytest.approx(-12.0)
    with pytest.raises(ValueError):
        semitones_between(0.0, 100.0)


def test_map_f0_to_target_register_preserves_contour():
    source = np.array([0.0, 90.0, 105.0, 120.0], dtype=np.float32)

    mapped, source_hz, shift = map_f0_to_target_register(source, 180.0)

    assert source_hz == pytest.approx(105.0)
    assert shift == pytest.approx(12 * np.log2(180 / 105))
    assert mapped[0] == 0.0
    np.testing.assert_allclose(
        mapped[1:] / mapped[2],
        source[1:] / source[2],
        rtol=1e-6,
    )
    assert mapped[2] == pytest.approx(180.0)


def test_map_f0_to_target_register_caps_extreme_shift():
    source = np.array([100.0, 110.0, 120.0], dtype=np.float32)

    mapped, _, shift = map_f0_to_target_register(
        source,
        400.0,
        max_shift_semitones=12.0,
    )

    assert shift == 12.0
    assert np.median(mapped) == pytest.approx(220.0)


@pytest.mark.parametrize("f0", [95.0, 190.0, 300.0])
def test_yin_tracker_recovers_register(f0):
    voiced = YinTracker(16000).push(harmonic_voice(f0))

    assert len(voiced) > 60
    assert np.median(voiced) == pytest.approx(f0, rel=0.015)


def test_yin_tracker_ignores_silence_and_noise():
    rng = np.random.default_rng(3)
    tracker = YinTracker(16000)

    assert len(tracker.push(np.zeros(16000, dtype=np.float32))) == 0
    assert len(tracker.push(0.0005 * rng.standard_normal(16000).astype(np.float32))) == 0
    noisy = tracker.push(0.1 * rng.standard_normal(32000).astype(np.float32))
    assert len(noisy) < 0.1 * (32000 / 320)


def test_yin_tracker_streaming_matches_single_push():
    signal = harmonic_voice(140.0, seconds=1.5)
    whole = YinTracker(16000).push(signal)
    streamed_tracker = YinTracker(16000)
    pieces = [
        streamed_tracker.push(signal[start : start + 1337])
        for start in range(0, len(signal), 1337)
    ]
    streamed = np.concatenate(pieces)

    np.testing.assert_allclose(streamed, whole)


def test_estimate_voiced_pitch_uses_voiced_frames_only():
    pause = np.zeros(8000, dtype=np.float32)
    audio = np.concatenate(
        [harmonic_voice(150.0, 1.2), pause, harmonic_voice(150.0, 1.2), pause]
    )

    pitch = estimate_voiced_pitch(audio, 16000)

    assert pitch.usable
    assert pitch.median_hz == pytest.approx(150.0, rel=0.03)
    assert pitch.p10_hz <= pitch.median_hz <= pitch.p90_hz


def test_estimate_voiced_pitch_rejects_noise():
    rng = np.random.default_rng(5)
    noise = 0.1 * rng.standard_normal(32000).astype(np.float32)

    assert not estimate_voiced_pitch(noise, 16000).usable


def test_register_mode_rejects_octave_cluster():
    rng = np.random.default_rng(7)
    true_register = 110.0 * 2.0 ** (rng.normal(0.0, 0.2, 200) / 12.0)
    octave_errors = 220.0 * 2.0 ** (rng.normal(0.0, 0.2, 60) / 12.0)

    mode = live_pitch._register_mode(np.concatenate([true_register, octave_errors]))

    assert mode == pytest.approx(110.0, rel=0.03)


def test_guard_stays_idle_when_output_matches_prompt():
    guard = RegisterGuard(prompt_hz=187.0)

    proposals = [guard.observe(np.full(60, 190.0)) for _ in range(6)]

    assert proposals == [None] * 6
    assert guard.state == "idle"
    assert guard.last_error == pytest.approx(0.27, abs=0.05)


def test_guard_never_acts_when_output_is_lower_than_prompt():
    guard = RegisterGuard(prompt_hz=170.0)

    assert [guard.observe(np.full(60, 100.0)) for _ in range(8)] == [None] * 8
    assert guard.state == "idle"


def test_guard_confirms_octave_error_then_locks_after_prompt_shift():
    guard = RegisterGuard(prompt_hz=110.0)

    first = guard.observe(np.full(60, 220.0))
    second = guard.observe(np.full(60, 220.0))

    assert first is None
    assert second == 2.0
    assert guard.state == "shifting"
    guard.applied(123.5)
    assert guard.observe(np.full(60, 122.0)) is None
    assert guard.state == "locked"
    assert guard.attempts == 1


def test_guard_walks_through_shifts_then_gives_up():
    guard = RegisterGuard(prompt_hz=110.0)
    proposed = []
    for _ in range(40):
        proposal = guard.observe(np.full(60, 440.0))
        if proposal is not None:
            proposed.append(proposal)
            guard.applied(110.0 * 2.0 ** (proposal / 12.0))

    assert proposed == list(live_pitch.GUARD_PROMPT_SHIFTS)
    assert guard.state == "failed"


def test_shift_pitch_preserves_length_and_scales_register():
    pytest.importorskip("parselmouth")
    sample_rate = 22050
    audio = harmonic_voice(120.0, seconds=2.0, sample_rate=sample_rate)

    shifted = shift_pitch_preserving_formants(audio, sample_rate, 3.0)

    assert len(shifted) == len(audio)
    assert np.max(np.abs(shifted)) <= 1.0
    before = estimate_voiced_pitch(audio, sample_rate)
    after = estimate_voiced_pitch(shifted, sample_rate)
    assert after.median_hz / before.median_hz == pytest.approx(2.0 ** (3.0 / 12.0), rel=0.04)


def test_shift_pitch_zero_returns_original_and_noise_is_rejected():
    audio = harmonic_voice(120.0, seconds=1.0)
    np.testing.assert_array_equal(shift_pitch_preserving_formants(audio, 16000, 0.0), audio)

    pytest.importorskip("parselmouth")
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError, match="no voiced speech"):
        shift_pitch_preserving_formants(
            0.05 * rng.standard_normal(16000).astype(np.float32), 16000, 2.0
        )


def test_voiced_pitch_usable_requires_confidence_and_duration():
    good = VoicedPitch(110.0, 100.0, 120.0, 1.5, 4.0, 0.9)
    assert good.usable
    assert not VoicedPitch(110.0, 100.0, 120.0, 1.5, 0.5, 0.9).usable
    assert not VoicedPitch(110.0, 100.0, 120.0, 1.5, 4.0, 0.2).usable
    assert not VoicedPitch(float("nan"), float("nan"), float("nan"), 0.0, 0.0, 0.0).usable
