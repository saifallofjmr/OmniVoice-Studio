# Design

## Direction

OmniVoice Studio is a compact native audio workbench. It uses familiar desktop
controls and a waveform-first hierarchy rather than a web dashboard, tab set,
or multi-step wizard. The full capture-to-clone workflow stays visible in one
workspace.

## Visual System

- **Surface:** Windows-native light utility chrome on `#f3f4f6`.
- **Primary action:** restrained system blue, reserved for focus and generation.
- **Recording:** red text communicates a live/destructive state without turning
  the whole interface into an alert.
- **Waveform:** deep slate canvas (`#111827`) with blue audio (`#60a5fa`),
  visible selection fill, boundary handles, centerline, and time ruler.
- **Status:** neutral slate, success green, and error red; every state is also
  written as text.

## Typography

Segoe UI is used throughout to match Windows. The 20 px semibold title anchors
the window; 10 px workhorse UI text carries controls and metadata. Measurements
remain compact and readable without using monospace as decoration.

## Layout

The window is a three-column command center that follows the task order:

1. **Left:** capture controls and a large waveform editor.
2. **Center:** Whisper transcript and text-clone generation.
3. **Right:** live-clone readiness, quality, routing, mixer gain, and meters.
4. **Bottom:** persistent global hotkeys and application status.

The waveform receives the largest share of space because audio selection is the
precision task. Transcript, text generation, live readiness, and routing remain
simultaneously visible so the user never has to navigate away from context.

Whisper transcription is visibly automatic but never treated as infallible:
the transcript remains a normal editable field, a manual transcribe action is
available, and status text asks the user to review the result.

## Interaction

- Dragging across the waveform creates a visible time selection.
- Playback uses the selection when present and the full clip otherwise.
- Keep and delete operations are explicit and undoable.
- Recording and model inference run outside the UI thread.
- Global hotkeys remain active when the window is minimized or unfocused and
  use no audible cue that could contaminate the loopback recording.
- Mixer meters show live microphone, AI injection, and final output levels;
  the virtual-route instruction stays visible beside mixer state.
- Live conversion is a deliberate mutually exclusive mode: starting it stops
  the normal mixer, releases other GPU models, and disables conflicting
  capture/generation actions until conversion ends.
- Quality is a named operational choice, not a cluster of model parameters:
  Fast, Balanced, Efficient fidelity, CS2 stable, CS2 survival, and High
  fidelity state the latency/similarity tradeoff while the engine receives the
  underlying prompt, guidance, step, and context values.
- Foreground-game overload is visible state: Studio explains the GPU conflict
  without changing quality, stopping the worker, or overriding the user's
  choice.
- Survival recovery is honest about degradation: converted audio may pause and
  accumulate delay, while status text explains that overload buffering is
  active and the raw microphone remains private.
- Hotkey detection uses a focused, temporary capture dialog because assigning
  a system-wide shortcut is a protected-focus task; the existing bindings are
  paused only for the duration of capture and restored on cancellation.
- Disabled states prevent conflicting capture, playback, editing, or generation.
- Escape stops active recording or playback; standard open, save, and undo
  shortcuts are supported.

## Accessibility

Controls use native focus behavior and keyboard traversal. Important state is
never color-only: recording time, selection bounds, duration, success, and
errors are stated in text. System dialogs handle file selection, and the layout
remains usable at the declared minimum window size.
