"""Capture an OmniVoice reference clip from an OS audio device.

Speaker capture uses the operating system's loopback device, so the resulting
WAV contains exactly what is playing through the selected output.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a voice-cloning reference WAV from speaker loopback.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available speaker output devices and exit.",
    )
    parser.add_argument(
        "--device",
        help="Speaker index, exact ID, or unique name substring. Uses the default speaker if omitted.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Recording duration in seconds. OmniVoice works best with 3-10 seconds.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Countdown before recording begins.",
    )
    parser.add_argument(
        "--capture-sample-rate",
        type=int,
        default=48000,
        help="Sample rate used while reading the OS loopback device.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Sample rate written to the reference WAV.",
    )
    parser.add_argument(
        "--output",
        default="reference.wav",
        help="Destination WAV path.",
    )
    parser.add_argument(
        "--transcript",
        help="Exact words spoken in the clip. Saved beside the WAV as a UTF-8 .txt file.",
    )
    return parser


def _load_soundcard():
    try:
        import soundcard as sc
    except ImportError as exc:
        raise RuntimeError(
            "Speaker capture requires SoundCard. Install it with "
            '`pip install "omnivoice[capture]"`.'
        ) from exc
    return sc


def _same_device(left, right) -> bool:
    return left is not None and right is not None and left.id == right.id


def _list_speakers(sc) -> list:
    speakers = list(sc.all_speakers())
    default = sc.default_speaker()
    if not speakers:
        raise RuntimeError("No speaker output devices were found.")

    for index, speaker in enumerate(speakers):
        marker = " (default)" if _same_device(speaker, default) else ""
        print(f"[{index}] {speaker.name}{marker}")
        print(f"    id: {speaker.id}")
    return speakers


def _resolve_speaker(sc, selector: str | None):
    speakers = list(sc.all_speakers())
    if not speakers:
        raise RuntimeError("No speaker output devices were found.")
    if selector is None:
        speaker = sc.default_speaker()
        if speaker is None:
            raise RuntimeError("No default speaker output device is configured.")
        return speaker

    try:
        index = int(selector)
    except ValueError:
        index = None
    if index is not None:
        if index < 0 or index >= len(speakers):
            raise ValueError(
                f"Speaker index {index} is out of range (0-{len(speakers) - 1})."
            )
        return speakers[index]

    exact_matches = [
        speaker
        for speaker in speakers
        if selector.casefold() in {speaker.id.casefold(), speaker.name.casefold()}
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    matches = [
        speaker
        for speaker in speakers
        if selector.casefold() in speaker.name.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No speaker device matches {selector!r}.")
    names = ", ".join(speaker.name for speaker in matches)
    raise ValueError(f"Speaker selector {selector!r} is ambiguous: {names}")


def _countdown(delay: float) -> None:
    whole_seconds = int(delay)
    remainder = delay - whole_seconds
    for remaining in range(whole_seconds, 0, -1):
        print(f"Recording in {remaining}...", flush=True)
        time.sleep(1)
    if remainder > 0:
        time.sleep(remainder)


def _prepare_audio(
    audio: np.ndarray, capture_sample_rate: int, output_sample_rate: int
) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if capture_sample_rate != output_sample_rate:
        import torch
        import torchaudio

        mono = (
            torchaudio.functional.resample(
                torch.from_numpy(mono),
                orig_freq=capture_sample_rate,
                new_freq=output_sample_rate,
            )
            .numpy()
            .astype(np.float32, copy=False)
        )
    return np.clip(mono, -1.0, 1.0)


def _validate_audio(audio: np.ndarray) -> tuple[float, float]:
    peak = float(np.max(np.abs(audio), initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    if peak < 1e-4 or rms < 1e-5:
        raise RuntimeError(
            "The recording is silent. Confirm audio is playing through the selected "
            "speaker and choose it with --device."
        )
    return peak, rms


def _capture(args) -> Path:
    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero.")
    if not 3 <= args.duration <= 10:
        print(
            "Warning: OmniVoice reference clips work best at 3-10 seconds.",
            file=sys.stderr,
        )
    if args.delay < 0:
        raise ValueError("--delay cannot be negative.")
    if args.capture_sample_rate <= 0 or args.sample_rate <= 0:
        raise ValueError("Sample rates must be greater than zero.")

    sc = _load_soundcard()
    speaker = _resolve_speaker(sc, args.device)
    loopback = sc.get_microphone(id=speaker.id, include_loopback=True)
    if loopback is None or not loopback.isloopback:
        raise RuntimeError(
            f"The OS did not expose a loopback recorder for {speaker.name!r}."
        )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = round(args.duration * args.capture_sample_rate)

    print(f"Device: {speaker.name}")
    print(f"Output: {output}")
    print("Play only the target voice during the recording window.")
    _countdown(args.delay)
    print(f"Recording for {args.duration:g} seconds...", flush=True)
    recorded = loopback.record(
        numframes=frame_count,
        samplerate=args.capture_sample_rate,
    )
    audio = _prepare_audio(recorded, args.capture_sample_rate, args.sample_rate)
    peak, rms = _validate_audio(audio)
    sf.write(output, audio, args.sample_rate, subtype="PCM_16")

    if args.transcript is not None:
        transcript_path = output.with_suffix(".txt")
        transcript_path.write_text(
            args.transcript.strip() + os.linesep, encoding="utf-8"
        )
        print(f"Transcript: {transcript_path}")

    print(
        f"Saved {len(audio) / args.sample_rate:.2f}s mono audio at "
        f"{args.sample_rate} Hz (peak={peak:.3f}, rms={rms:.3f})."
    )
    if peak >= 0.999:
        print(
            "Warning: the capture reached full scale and may be clipped.",
            file=sys.stderr,
        )
    elif peak < 0.02 or rms < 0.003:
        print(
            "Warning: the capture level is very low. Raise the speaker volume and "
            "record again for a better cloning reference.",
            file=sys.stderr,
        )
    return output


def main() -> None:
    args = get_parser().parse_args()
    try:
        sc = _load_soundcard()
        if args.list_devices:
            _list_speakers(sc)
            return
        _capture(args)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
