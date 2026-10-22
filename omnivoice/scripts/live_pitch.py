"""Pitch and register utilities shared by Studio and the Seed-VC live worker.

Only numpy and librosa are required at import time (Praat is imported lazily),
so both the Studio (Python 3.12) and the worker (Python 3.10) can use this file.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

ANALYSIS_RATE = 16000
PYIN_FRAME_LENGTH = 1024
PYIN_HOP_LENGTH = 160
MIN_F0_HZ = 60.0
MAX_F0_HZ = 520.0

# Seed-VC Tiny sometimes renders low-pitched prompts an octave too high. Output
# register follows the prompt register, so a small formant-preserving prompt
# shift moves the model out of the failing region.
GUARD_ERROR_THRESHOLD_SEMITONES = 6.0
GUARD_PROMPT_SHIFTS = (2.0, 3.0, 1.5, 4.0)
MAX_MANUAL_PITCH_SEMITONES = 6.0


@dataclass(frozen=True)
class VoicedPitch:
    median_hz: float
    p10_hz: float
    p90_hz: float
    std_semitones: float
    voiced_seconds: float
    confidence: float

    @property
    def usable(self) -> bool:
        return (
            math.isfinite(self.median_hz)
            and self.voiced_seconds >= 1.5
            and self.confidence >= 0.45
        )


def semitones_between(reference_hz: float, measured_hz: float) -> float:
    if reference_hz <= 0 or measured_hz <= 0:
        raise ValueError("Pitch values must be positive.")
    return 12.0 * math.log2(measured_hz / reference_hz)


def map_f0_to_target_register(
    f0_hz: np.ndarray,
    target_hz: float,
    *,
    max_shift_semitones: float = 12.0,
) -> tuple[np.ndarray, float, float]:
    """Shift voiced F0 around its median while preserving its relative contour."""
    f0 = np.asarray(f0_hz, dtype=np.float32).copy()
    voiced = f0[(f0 >= 55.0) & (f0 <= 600.0)]
    if voiced.size == 0 or not math.isfinite(target_hz) or target_hz <= 0:
        return f0, float("nan"), 0.0
    source_hz = float(np.median(voiced))
    shift = semitones_between(source_hz, target_hz)
    shift = max(-max_shift_semitones, min(max_shift_semitones, shift))
    f0[f0 > 0] *= 2.0 ** (shift / 12.0)
    return f0, source_hz, shift


def _register_mode(f0_hz: np.ndarray) -> float:
    # Octave errors split voiced frames into separated clusters; the densest
    # quarter-semitone bin identifies the real register.
    semitones = 12.0 * np.log2(f0_hz / 100.0)
    bins = np.arange(semitones.min() - 1.0, semitones.max() + 1.25, 0.25)
    counts, edges = np.histogram(semitones, bins=bins)
    kernel = np.hanning(9)
    kernel /= kernel.sum()
    smoothed = np.convolve(counts.astype(np.float64), kernel, mode="same")
    peak = int(np.argmax(smoothed))
    centre = 0.5 * (edges[peak] + edges[peak + 1])
    return float(100.0 * 2.0 ** (centre / 12.0))


def estimate_voiced_pitch(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fmin: float = MIN_F0_HZ,
    fmax: float = MAX_F0_HZ,
) -> VoicedPitch:
    """Median F0 over energetic, confidently voiced frames with octave rejection."""
    import librosa

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Cannot estimate pitch from empty audio.")
    if sample_rate != ANALYSIS_RATE:
        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=ANALYSIS_RATE
        )
    audio = audio - float(np.mean(audio))
    f0, voiced_flag, _voiced_prob = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=ANALYSIS_RATE,
        frame_length=PYIN_FRAME_LENGTH,
        hop_length=PYIN_HOP_LENGTH,
        fill_na=np.nan,
    )
    rms = librosa.feature.rms(
        y=audio,
        frame_length=PYIN_FRAME_LENGTH,
        hop_length=PYIN_HOP_LENGTH,
        center=True,
    )[0]
    count = min(len(f0), len(rms))
    f0, voiced_flag, rms = f0[:count], voiced_flag[:count], rms[:count]
    energetic = rms >= 0.03 * float(np.max(rms, initial=0.0))
    candidates = voiced_flag & energetic & np.isfinite(f0)
    unusable = VoicedPitch(float("nan"), float("nan"), float("nan"), 0.0, 0.0, 0.0)
    if np.count_nonzero(candidates) < 20:
        return unusable

    register = _register_mode(f0[candidates])
    offsets = np.abs(12.0 * np.log2(np.where(candidates, f0, register) / register))
    kept = candidates & (offsets <= 5.0)
    voiced = f0[kept]
    centre = float(np.median(voiced))
    kept_fraction = voiced.size / max(1, int(np.count_nonzero(candidates)))
    voiced_fraction = voiced.size / max(1, int(np.count_nonzero(energetic)))
    confidence = float(
        np.clip(kept_fraction * min(1.0, voiced_fraction / 0.5), 0.0, 1.0)
    )
    return VoicedPitch(
        median_hz=centre,
        p10_hz=float(np.percentile(voiced, 10)),
        p90_hz=float(np.percentile(voiced, 90)),
        std_semitones=float(np.std(12.0 * np.log2(voiced / centre))),
        voiced_seconds=float(voiced.size * PYIN_HOP_LENGTH / ANALYSIS_RATE),
        confidence=confidence,
    )


class YinTracker:
    """Light streaming YIN tracker returning F0 for voiced frames only."""

    def __init__(
        self,
        sample_rate: int = ANALYSIS_RATE,
        *,
        fmin: float = 65.0,
        fmax: float = 500.0,
        hop_seconds: float = 0.02,
        threshold: float = 0.15,
        voiced_ceiling: float = 0.25,
        rms_floor: float = 0.003,
    ):
        self.sample_rate = sample_rate
        self.tau_min = max(2, int(sample_rate / fmax))
        self.tau_max = int(sample_rate / fmin) + 1
        self.window = 2 * self.tau_max
        self.frame_length = self.window + self.tau_max
        self.hop = int(round(sample_rate * hop_seconds))
        self.threshold = threshold
        self.voiced_ceiling = voiced_ceiling
        self.rms_floor = rms_floor
        self._fft_size = 1 << int(math.ceil(math.log2(self.frame_length + self.window)))
        self._buffer = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> np.ndarray:
        self._buffer = np.concatenate(
            (self._buffer, np.asarray(samples, dtype=np.float32).reshape(-1))
        )
        if len(self._buffer) < self.frame_length:
            return np.empty(0, dtype=np.float64)
        count = 1 + (len(self._buffer) - self.frame_length) // self.hop
        starts = np.arange(count) * self.hop
        index = starts[:, None] + np.arange(self.frame_length)[None, :]
        frames = self._buffer[index].astype(np.float64)
        self._buffer = self._buffer[count * self.hop :]
        return self._voiced_f0(frames)

    def _voiced_f0(self, frames: np.ndarray) -> np.ndarray:
        window = self.window
        tau_max = self.tau_max
        energy = np.cumsum(frames**2, axis=1)
        energy = np.concatenate((np.zeros((len(frames), 1)), energy), axis=1)
        head = energy[:, window] - energy[:, 0]
        lagged = energy[:, window : window + tau_max + 1] - energy[:, : tau_max + 1]
        spectrum = np.fft.rfft(frames, n=self._fft_size, axis=1)
        reference = np.fft.rfft(frames[:, :window], n=self._fft_size, axis=1)
        cross = np.fft.irfft(spectrum * np.conj(reference), n=self._fft_size, axis=1)[
            :, : tau_max + 1
        ]
        difference = head[:, None] + lagged - 2.0 * cross
        difference[:, 0] = 0.0
        running = np.cumsum(difference[:, 1:], axis=1)
        cmndf = np.ones_like(difference)
        lags = np.arange(1, tau_max + 1)[None, :]
        cmndf[:, 1:] = difference[:, 1:] * lags / np.maximum(running, 1e-12)

        rms = np.sqrt(head / window)
        voiced_f0 = []
        band = cmndf[:, self.tau_min : tau_max + 1]
        below = band < self.threshold
        for row in range(len(frames)):
            if rms[row] < self.rms_floor:
                continue
            if below[row].any():
                tau = int(np.argmax(below[row])) + self.tau_min
                while tau + 1 <= tau_max and cmndf[row, tau + 1] < cmndf[row, tau]:
                    tau += 1
            else:
                tau = int(np.argmin(band[row])) + self.tau_min
            if cmndf[row, tau] > self.voiced_ceiling or tau <= 1 or tau >= tau_max:
                continue
            left, centre, right = cmndf[row, tau - 1], cmndf[row, tau], cmndf[row, tau + 1]
            denominator = left - 2.0 * centre + right
            refined = tau + (0.5 * (left - right) / denominator if denominator > 1e-12 else 0.0)
            voiced_f0.append(self.sample_rate / refined)
        return np.asarray(voiced_f0, dtype=np.float64)


@dataclass
class RegisterGuard:
    """Detects octave-up register failures and proposes prompt pitch shifts."""

    prompt_hz: float
    threshold_semitones: float = GUARD_ERROR_THRESHOLD_SEMITONES
    shifts: tuple[float, ...] = GUARD_PROMPT_SHIFTS
    evaluate_every: int = 50
    window_frames: int = 150
    confirmations: int = 2
    attempts: int = 0
    strikes: int = 0
    last_error: float | None = None
    state: str = "idle"
    _history: deque = field(default_factory=lambda: deque(maxlen=400))
    _since_evaluation: int = 0

    def observe(self, voiced_f0: np.ndarray) -> float | None:
        """Feed voiced output F0 frames; return a prompt shift (semitones) to try."""
        if self.state == "failed" or not len(voiced_f0):
            return None
        self._history.extend(float(value) for value in voiced_f0)
        self._since_evaluation += len(voiced_f0)
        if self._since_evaluation < self.evaluate_every or len(self._history) < self.evaluate_every:
            return None
        self._since_evaluation = 0
        recent = list(self._history)[-self.window_frames :]
        self.last_error = semitones_between(self.prompt_hz, float(np.median(recent)))
        if self.last_error <= self.threshold_semitones:
            self.strikes = 0
            if self.attempts:
                self.state = "locked"
            return None
        self.strikes += 1
        if self.strikes < self.confirmations:
            return None
        if self.attempts >= len(self.shifts):
            self.state = "failed"
            return None
        shift = self.shifts[self.attempts]
        self.attempts += 1
        self.state = "shifting"
        self.strikes = 0
        return shift

    def applied(self, new_prompt_hz: float) -> None:
        self.prompt_hz = new_prompt_hz
        self._history.clear()
        self._since_evaluation = 0
        self.strikes = 0
        self.last_error = None
        self.state = "checking"


def shift_pitch_preserving_formants(
    audio: np.ndarray,
    sample_rate: int,
    semitones: float,
    *,
    fmin: float = MIN_F0_HZ,
    fmax: float = MAX_F0_HZ,
) -> np.ndarray:
    """Praat PSOLA pitch shift that keeps formants and the original length."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if abs(semitones) < 1e-6:
        return audio
    import parselmouth
    from parselmouth.praat import call

    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sample_rate)
    pitch = sound.to_pitch_ac(pitch_floor=fmin, pitch_ceiling=fmax)
    median = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")
    if not (median and math.isfinite(median) and median > 0):
        raise ValueError("The reference has no voiced speech to shift.")
    shifted = call(
        sound,
        "Change gender",
        fmin,
        fmax,
        1.0,
        median * 2.0 ** (semitones / 12.0),
        1.0,
        1.0,
    )
    result = np.asarray(shifted.values[0], dtype=np.float32)
    # Identical length keeps the compiled estimator's tensor shapes unchanged.
    if len(result) < len(audio):
        result = np.pad(result, (0, len(audio) - len(result)))
    result = result[: len(audio)]
    return np.ascontiguousarray(np.clip(result, -1.0, 1.0), dtype=np.float32)
