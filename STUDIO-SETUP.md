# OmniVoice Studio for Windows

This repository extends [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)
with a native Windows workflow for:

- speaker-loopback recording and waveform trimming;
- automatic Whisper transcription;
- OmniVoice text-to-speech cloning;
- configurable global hotkeys;
- live microphone + generated-clip mixing through VoiceMeeter;
- zero-shot live voice conversion through an isolated Seed-VC realtime engine.

## Quick setup

Requirements:

- Windows 10/11 with `winget`;
- an NVIDIA GPU compatible with CUDA 12.8;
- substantial free disk space for PyTorch and public model checkpoints.

Clone the repository, then double-click:

```text
setup-and-launch.cmd
```

The script installs Python, Git, FFmpeg, VoiceMeeter, CUDA PyTorch,
OmniVoice dependencies, public checkpoints, and the isolated Seed-VC engine.
No API token or private credential is required.

If VoiceMeeter is installed for the first time, restart Windows, then run:

```text
launch-studio.cmd
```

## CS2 routing

1. In Studio, select the physical microphone and `Voicemeeter Input`.
2. Click **Start mixer** for normal mic + generated clips, or record/open a
   reference and click **Start live clone** for realtime conversion.
3. In CS2, select **Voicemeeter Out 6 (VB-Audio Voicemeeter VAIO)**.

**Send latest clip to CS2 mixer** works in either mode. During Live Clone, the
clip uses a separate generated-audio-only VoiceMeeter stream, so the physical
microphone is never passed through unconverted.

## Live clone quality

- **Fast:** lower latency, weaker similarity.
- **Balanced:** moderate latency and identity.
- **Enhanced fidelity:** stronger identity and streaming continuity while
  retaining more gaming headroom than High fidelity.
- **v0.3.0 legacy:** restores the original raw-reference/eager live path when
  newer conditioning sounds artificial on a particular voice.
- **High fidelity:** strongest zero-shot similarity, highest latency.

Use 6-15 seconds of clean, dry, authorized speech for the best result.
Studio automatically removes silent reference edges and normalizes the
conditioning level used by the live engine.

## Responsible use

Only clone or convert toward voices you own or have permission to use. Do not
use this software for impersonation, fraud, harassment, or evading moderation.

## Third-party software

- OmniVoice is provided under the repository's Apache-2.0 license.
- Seed-VC realtime is cloned separately at setup time and remains subject to
  its GPL-3.0 license.
- VoiceMeeter and FFmpeg are installed from their respective distributors and
  remain subject to their own licenses.
