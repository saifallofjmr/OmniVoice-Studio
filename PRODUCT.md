# Product

<!-- impeccable:product-schema 1 -->

## Platform

adaptive

## Users

OmniVoice users working locally on a desktop who need to capture authorized
speech playing through an operating-system audio device, clean the clip, and
use it as a zero-shot voice-cloning reference.

## Product Purpose

Provide a compact native desktop workflow for recording, reviewing, trimming,
saving, and immediately testing an OmniVoice reference clip without requiring
command-line audio editing.

## Positioning

The app connects operating-system speaker loopback directly to OmniVoice's
reference-audio and generation workflow, keeping capture, waveform editing,
transcription, and clone testing in one local tool.

## Operating Context

The primary environment is Windows with a local NVIDIA GPU. Users play a clean
target-voice sample through a selected speaker device, stop recording, select
the useful waveform region, trim it, enter the exact transcript, and generate a
test sentence. Other system audio, calls, games, and notifications must be
muted during capture.

## Capabilities and Constraints

- Records system speaker loopback from a selectable operating-system device.
- Provides configurable persistent global start/stop hotkeys for recording
  while a game or another application has focus.
- Mixes the selected physical microphone with queued OmniVoice speech and
  sends the limited result to a selected virtual microphone feed.
- Converts the selected physical microphone in real time toward a 1-30 second
  authorized reference using an isolated zero-shot Seed-VC engine.
- Provides a global play hotkey for injecting the latest generated clip while
  a game has focus.
- Supports record, stop, playback, waveform selection, trim, undo, open, save,
  automatic Whisper transcription with manual correction, and cloned-speech
  generation.
- Produces mono 24 kHz WAV references suited to OmniVoice.
- Reference clips work best when they contain 3-10 seconds of one clean,
  authorized voice with an exact transcript.
- Model inference can take time and must run off the UI thread.
- Global hotkeys use the Windows operating-system registration API and do not
  inject or synthesize keyboard input.
- In-game microphone routing requires a signed virtual audio endpoint such as
  VoiceMeeter; with the current driver, CS2 must select `Voicemeeter Out 6`,
  which carries the configured B1 bus.
- Live conversion has perceptible latency and releases the Studio Whisper/TTS
  models before startup to preserve GPU headroom while gaming.
- Live conversion exposes fast, balanced, and high-fidelity presets that trade
  latency for prompt duration, similarity guidance, diffusion steps, and
  context.
- An efficient-fidelity preset keeps strong CFG guidance while shortening the
  prompt and reducing diffusion work for gaming under shared GPU load.
- An underrun monitor reports foreground-game GPU starvation without changing
  the user's selected quality preset.
- Under extreme overload, CS2-survival mode preserves delayed converted audio
  in an expanded buffer. Output may pause or gain latency, but the unmodified
  microphone is never exposed.
- The initial implementation is a local Python desktop app using native desktop
  controls, not a hosted web interface.

## Evidence on Hand

- Working OmniVoice installation and local checkpoint at `local/OmniVoice`.
- Verified Windows speaker loopback capture through SoundCard.
- Verified CUDA inference on an NVIDIA GeForce RTX 5080.
- No user recordings or voice assets are bundled with the product.

## Product Principles

- Keep the reference-clipping workflow visible and direct.
- Never hide recording, inference, or device errors.
- Preserve audio locally and require deliberate save/export actions.
- Favor clean voice data over automated enhancement that could alter identity.
- Make recording state and selected clip boundaries unmistakable.

## Accessibility & Inclusion

Core actions must be keyboard reachable, state must not rely on color alone,
and timing or duration values must be available as text in addition to the
waveform.
