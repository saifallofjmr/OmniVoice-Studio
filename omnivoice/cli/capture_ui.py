"""Native reference-recording and voice-cloning studio.

THESIS: A waveform-first desktop tool that refuses the browser-dashboard
pattern and keeps capture, clipping, and clone testing in one visible flow.
OWN-WORLD: Windows-native controls, cool neutral surfaces, one blue action
color, a dark precision waveform, and red reserved for live recording.
STORY: Choose an output, record clean speech, shape the useful clip, enter its
exact words, then prove the reference by generating a test line.
FIRST VIEWPORT: Device and transport controls lead directly into a large
waveform editor; transcript and clone testing occupy the lower workbench.
FORM: Compact single-window audio workbench, the first and most direct
structure for a frequent operate-mode task.
"""

import argparse
import ctypes
import gc
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import soundfile as sf

from omnivoice.cli.capture import (
    _load_soundcard,
    _prepare_audio,
    _validate_audio,
)

REFERENCE_SAMPLE_RATE = 24000
CAPTURE_SAMPLE_RATE = 48000
MAX_RECORDING_SECONDS = 120
PLAYBACK_BLOCK_SIZE = 2048
HOTKEY_CONFIG_PATH = (
    Path(os.environ.get("APPDATA", Path.home())) / "OmniVoice" / "studio.json"
)
DEFAULT_RECORD_HOTKEY = "Ctrl+Alt+F9"
DEFAULT_STOP_HOTKEY = "Ctrl+Alt+F10"
DEFAULT_PLAY_HOTKEY = "Ctrl+Alt+F11"
MIXER_SAMPLE_RATE = 48000
MIXER_BLOCK_SIZE = 480
VOICEMEETER_DIRECTORY = Path(r"C:\Program Files (x86)\VB\Voicemeeter")
VOICEMEETER_REMOTE_DLL = VOICEMEETER_DIRECTORY / "VoicemeeterRemote64.dll"
SEED_VC_DIRECTORY = Path(os.environ.get("SEED_VC_HOME", Path.home() / "SeedVC"))
SEED_VC_PYTHON = SEED_VC_DIRECTORY / ".venv" / "Scripts" / "python.exe"
SEED_VC_REFERENCE = SEED_VC_DIRECTORY / "references" / "studio-reference.wav"
SEED_VC_LOG = SEED_VC_DIRECTORY / "live-voice.log"
LIVE_VOICE_WORKER = (
    Path(__file__).resolve().parents[1] / "scripts" / "live_voice_worker.py"
)
LIVE_VOICE_PRESETS = {
    "Fast": {
        "diffusion_steps": 4,
        "inference_cfg_rate": 0.0,
        "max_prompt_length": 3.0,
        "block_time": 0.18,
        "extra_time_ce": 2.5,
        "extra_time": 0.5,
    },
    "Balanced": {
        "diffusion_steps": 8,
        "inference_cfg_rate": 0.4,
        "max_prompt_length": 6.0,
        "block_time": 0.28,
        "extra_time_ce": 3.0,
        "extra_time": 0.7,
    },
    "CS2 stable": {
        "diffusion_steps": 8,
        "inference_cfg_rate": 0.0,
        "max_prompt_length": 6.0,
        "block_time": 0.3,
        "extra_time_ce": 2.5,
        "extra_time": 0.5,
    },
    "CS2 survival": {
        "diffusion_steps": 2,
        "inference_cfg_rate": 0.0,
        "max_prompt_length": 3.0,
        "block_time": 0.4,
        "extra_time_ce": 2.0,
        "extra_time": 0.5,
    },
    "High fidelity": {
        "diffusion_steps": 12,
        "inference_cfg_rate": 0.7,
        "max_prompt_length": 10.0,
        "block_time": 0.4,
        "extra_time_ce": 4.0,
        "extra_time": 1.0,
    },
}

_MODIFIERS = {
    "alt": (0x0001, "Alt"),
    "ctrl": (0x0002, "Ctrl"),
    "control": (0x0002, "Ctrl"),
    "shift": (0x0004, "Shift"),
    "win": (0x0008, "Win"),
    "windows": (0x0008, "Win"),
}
_NAMED_KEYS = {
    "space": (0x20, "Space"),
    "pageup": (0x21, "PageUp"),
    "pagedown": (0x22, "PageDown"),
    "end": (0x23, "End"),
    "home": (0x24, "Home"),
    "left": (0x25, "Left"),
    "up": (0x26, "Up"),
    "right": (0x27, "Right"),
    "down": (0x28, "Down"),
    "insert": (0x2D, "Insert"),
    "delete": (0x2E, "Delete"),
    "pause": (0x13, "Pause"),
    "scrolllock": (0x91, "ScrollLock"),
}
_TK_MODIFIER_KEYSYMS = {
    "Control_L": "Ctrl",
    "Control_R": "Ctrl",
    "Alt_L": "Alt",
    "Alt_R": "Alt",
    "Shift_L": "Shift",
    "Shift_R": "Shift",
    "Super_L": "Win",
    "Super_R": "Win",
    "Win_L": "Win",
    "Win_R": "Win",
}
_TK_NAMED_KEYSYMS = {
    "space": "Space",
    "Prior": "PageUp",
    "Next": "PageDown",
    "End": "End",
    "Home": "Home",
    "Left": "Left",
    "Up": "Up",
    "Right": "Right",
    "Down": "Down",
    "Insert": "Insert",
    "Delete": "Delete",
    "Pause": "Pause",
    "Scroll_Lock": "ScrollLock",
}


@dataclass(frozen=True)
class HotkeySpec:
    text: str
    modifiers: int
    virtual_key: int


@dataclass(frozen=True)
class PortAudioDevice:
    index: int
    name: str


def parse_hotkey(value: str) -> HotkeySpec:
    tokens = [token.strip() for token in value.split("+") if token.strip()]
    if not tokens:
        raise ValueError("Enter a hotkey such as Ctrl+Alt+F9.")

    modifiers = 0
    key = None
    key_name = None
    for token in tokens:
        normalized = token.casefold().replace(" ", "")
        if normalized in _MODIFIERS:
            modifier, _name = _MODIFIERS[normalized]
            if not modifiers & modifier:
                modifiers |= modifier
            continue
        if key is not None:
            raise ValueError("A hotkey can contain only one non-modifier key.")
        if len(normalized) == 1 and normalized.isalnum():
            key_name = normalized.upper()
            key = ord(key_name)
        elif normalized.startswith("f") and normalized[1:].isdigit():
            function_number = int(normalized[1:])
            if not 1 <= function_number <= 24:
                raise ValueError("Function keys must be between F1 and F24.")
            key_name = f"F{function_number}"
            key = 0x70 + function_number - 1
        elif normalized in _NAMED_KEYS:
            key, key_name = _NAMED_KEYS[normalized]
        else:
            raise ValueError(f"Unsupported hotkey key: {token!r}.")

    if key is None:
        raise ValueError("The hotkey is missing its main key.")
    canonical_modifiers = [
        name
        for bit, name in (
            (0x0002, "Ctrl"),
            (0x0001, "Alt"),
            (0x0004, "Shift"),
            (0x0008, "Win"),
        )
        if modifiers & bit
    ]
    return HotkeySpec(
        text="+".join((*canonical_modifiers, key_name)),
        modifiers=modifiers,
        virtual_key=key,
    )


def detected_hotkey(modifiers: set[str], keysym: str) -> HotkeySpec:
    is_alphanumeric = len(keysym) == 1 and keysym.isalnum()
    is_function_key = keysym.casefold().startswith("f") and keysym[1:].isdigit()
    if is_alphanumeric or is_function_key:
        key_name = keysym.upper()
    elif keysym in _TK_NAMED_KEYSYMS:
        key_name = _TK_NAMED_KEYSYMS[keysym]
    else:
        raise ValueError(f"Unsupported key: {keysym}.")
    return parse_hotkey("+".join((*modifiers, key_name)))


class GlobalHotkeyManager:
    WM_HOTKEY = 0x0312
    WM_QUIT = 0x0012
    MOD_NOREPEAT = 0x4000

    def __init__(self, event_queue: queue.Queue):
        self.event_queue = event_queue
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._ready = threading.Event()
        self._startup_error: str | None = None

    def start(self, bindings: dict[str, HotkeySpec]) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Global hotkeys currently require Windows.")
        self.stop()
        self._ready.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._run,
            args=(bindings.copy(),),
            daemon=True,
            name="OmniVoiceGlobalHotkeys",
        )
        self._thread.start()
        if not self._ready.wait(timeout=3):
            raise RuntimeError("Timed out while registering global hotkeys.")
        if self._startup_error:
            raise RuntimeError(self._startup_error)

    def _run(self, bindings: dict[str, HotkeySpec]) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._thread_id = kernel32.GetCurrentThreadId()

        message = wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(message), None, 0, 0, 0)
        registered_ids = []
        actions = {}
        try:
            for hotkey_id, (action, spec) in enumerate(bindings.items(), start=1):
                registered = user32.RegisterHotKey(
                    None,
                    hotkey_id,
                    spec.modifiers | self.MOD_NOREPEAT,
                    spec.virtual_key,
                )
                if not registered:
                    error_code = ctypes.get_last_error()
                    raise RuntimeError(
                        f"Could not register {spec.text} for {action} "
                        f"(Windows error {error_code}). It may already be in use."
                    )
                registered_ids.append(hotkey_id)
                actions[hotkey_id] = action
            self._ready.set()

            while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                if message.message == self.WM_HOTKEY:
                    action = actions.get(int(message.wParam))
                    if action is not None:
                        self.event_queue.put(("hotkey", action))
        except RuntimeError as exc:
            self._startup_error = str(exc)
            self._ready.set()
        finally:
            for hotkey_id in registered_ids:
                user32.UnregisterHotKey(None, hotkey_id)
            self._thread_id = None

    def stop(self) -> None:
        if self._thread is None:
            return
        if self._thread.is_alive() and self._thread_id is not None:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, self.WM_QUIT, 0, 0)
            self._thread.join(timeout=2)
        self._thread = None
        self._thread_id = None


def mix_audio_block(
    microphone: np.ndarray,
    generated: np.ndarray,
    microphone_gain: float,
    generated_gain: float,
) -> np.ndarray:
    if microphone.shape != generated.shape:
        raise ValueError("Mixer inputs must have matching shapes.")
    mixed = (
        np.asarray(microphone, dtype=np.float32) * microphone_gain
        + np.asarray(generated, dtype=np.float32) * generated_gain
    )
    peak = float(np.max(np.abs(mixed), initial=0.0))
    if peak > 0.98:
        mixed *= 0.98 / peak
    return mixed.astype(np.float32, copy=False)


def configure_voicemeeter_b1() -> None:
    if sys.platform != "win32":
        raise RuntimeError("VoiceMeeter routing currently requires Windows.")
    if not VOICEMEETER_REMOTE_DLL.exists():
        raise RuntimeError(
            "VoiceMeeter is not installed. Install VB-Audio VoiceMeeter first."
        )

    remote = ctypes.WinDLL(str(VOICEMEETER_REMOTE_DLL))
    remote.VBVMR_Login.restype = ctypes.c_long
    remote.VBVMR_Logout.restype = ctypes.c_long
    remote.VBVMR_RunVoicemeeter.argtypes = [ctypes.c_long]
    remote.VBVMR_RunVoicemeeter.restype = ctypes.c_long
    remote.VBVMR_SetParameterFloat.argtypes = [ctypes.c_char_p, ctypes.c_float]
    remote.VBVMR_SetParameterFloat.restype = ctypes.c_long

    login_result = remote.VBVMR_Login()
    if login_result == 1:
        remote.VBVMR_RunVoicemeeter(1)
        remote.VBVMR_Logout()
        time.sleep(2)
        login_result = remote.VBVMR_Login()
    if login_result != 0:
        raise RuntimeError(
            f"Could not connect to VoiceMeeter (API error {login_result})."
        )

    try:
        parameters = {
            b"Strip[2].B1": 1.0,
            b"Strip[2].A1": 0.0,
            b"Strip[2].Mute": 0.0,
            b"Bus[1].Mute": 0.0,
        }
        for name, value in parameters.items():
            result = remote.VBVMR_SetParameterFloat(name, ctypes.c_float(value))
            if result != 0:
                raise RuntimeError(
                    f"VoiceMeeter rejected {name.decode()} (API error {result})."
                )
    finally:
        remote.VBVMR_Logout()


def is_virtual_output_name(name: str) -> bool:
    normalized = name.casefold()
    return (
        normalized.startswith("voicemeeter input")
        or "cable input" in normalized
        or "virtual" in normalized
        or "sonar - stream" in normalized
    )


def is_virtual_audio_device_name(name: str) -> bool:
    normalized = name.casefold()
    return any(
        marker in normalized
        for marker in ("voicemeeter", "vb-audio cable", "virtual", "sonar")
    )


def is_physical_microphone_name(name: str) -> bool:
    return not is_virtual_audio_device_name(name)


class RealtimeMicMixer:
    def __init__(self, event_queue: queue.Queue):
        self.event_queue = event_queue
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None
        self._pending_generated = np.empty(0, dtype=np.float32)
        self._microphone_gain = 1.0
        self._generated_gain = 1.0
        self._gain_lock = threading.Lock()
        self._meter_counter = 0
        self._last_stream_status = ""

    @property
    def running(self) -> bool:
        return self._stream is not None and bool(self._stream.active)

    def set_gains(self, microphone_gain: float, generated_gain: float) -> None:
        with self._gain_lock:
            self._microphone_gain = max(0.0, min(2.0, microphone_gain))
            self._generated_gain = max(0.0, min(2.0, generated_gain))

    def queue_generated(self, audio: np.ndarray) -> None:
        if not self.running:
            raise RuntimeError("Start the game microphone mixer first.")
        resampled = _prepare_audio(
            np.asarray(audio, dtype=np.float32)[:, np.newaxis],
            REFERENCE_SAMPLE_RATE,
            MIXER_SAMPLE_RATE,
        )
        self._audio_queue.put(resampled)

    def start(self, microphone_device: int, virtual_output_device: int) -> None:
        if self.running:
            raise RuntimeError("The game microphone mixer is already running.")
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        self._pending_generated = np.empty(0, dtype=np.float32)
        self._meter_counter = 0
        self._last_stream_status = ""

        import sounddevice as sd

        try:
            self._stream = sd.Stream(
                device=(microphone_device, virtual_output_device),
                samplerate=MIXER_SAMPLE_RATE,
                blocksize=MIXER_BLOCK_SIZE,
                dtype="float32",
                channels=(1, 2),
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            self._stream = None
            raise RuntimeError(f"Could not open the audio mixer: {exc}") from exc
        self.event_queue.put(("mixer_started",))

    def stop(self) -> None:
        if self._stream is None:
            return
        import sounddevice as sd

        had_error = False
        try:
            self._stream.stop()
            self._stream.close()
        except sd.PortAudioError as exc:
            had_error = True
            self.event_queue.put(("mixer_error", str(exc)))
        finally:
            self._stream = None
        self.event_queue.put(("mixer_stopped", had_error))

    def _callback(self, indata, outdata, frames, _time_info, status) -> None:
        if status:
            status_text = str(status)
            if status_text != self._last_stream_status:
                self._last_stream_status = status_text
                self.event_queue.put(("mixer_warning", status_text))

        microphone_block = np.asarray(indata[:, 0], dtype=np.float32)
        while len(self._pending_generated) < frames:
            try:
                next_audio = self._audio_queue.get_nowait()
            except queue.Empty:
                break
            self._pending_generated = np.concatenate(
                (self._pending_generated, next_audio)
            )

        generated_block = np.zeros(frames, dtype=np.float32)
        generated_count = min(frames, len(self._pending_generated))
        if generated_count:
            generated_block[:generated_count] = self._pending_generated[
                :generated_count
            ]
            self._pending_generated = self._pending_generated[generated_count:]

        with self._gain_lock:
            microphone_gain = self._microphone_gain
            generated_gain = self._generated_gain
        mixed = mix_audio_block(
            microphone_block,
            generated_block,
            microphone_gain,
            generated_gain,
        )
        outdata[:, 0] = mixed
        outdata[:, 1] = mixed

        self._meter_counter += 1
        if self._meter_counter >= 10:
            self._meter_counter = 0
            microphone_level = float(
                np.sqrt(np.mean(np.square(microphone_block), dtype=np.float64))
            )
            generated_level = float(
                np.sqrt(np.mean(np.square(generated_block), dtype=np.float64))
            )
            output_level = float(np.sqrt(np.mean(np.square(mixed), dtype=np.float64)))
            self.event_queue.put(
                (
                    "mixer_levels",
                    microphone_level,
                    generated_level,
                    output_level,
                )
            )


def _selection_indices(
    audio_length: int,
    sample_rate: int,
    selection: tuple[float, float] | None,
) -> tuple[int, int]:
    if selection is None:
        return 0, audio_length
    start, end = sorted(selection)
    start_index = max(0, min(audio_length, round(start * sample_rate)))
    end_index = max(0, min(audio_length, round(end * sample_rate)))
    return start_index, end_index


def keep_selection(
    audio: np.ndarray,
    sample_rate: int,
    selection: tuple[float, float] | None,
) -> np.ndarray:
    start, end = _selection_indices(len(audio), sample_rate, selection)
    if end <= start:
        raise ValueError("Drag across the waveform to select audio first.")
    return audio[start:end].copy()


def delete_selection(
    audio: np.ndarray,
    sample_rate: int,
    selection: tuple[float, float] | None,
) -> np.ndarray:
    start, end = _selection_indices(len(audio), sample_rate, selection)
    if selection is None or end <= start:
        raise ValueError("Drag across the waveform to select audio first.")
    result = np.concatenate((audio[:start], audio[end:]))
    if not len(result):
        raise ValueError("Deleting the selection would remove the entire clip.")
    return result


def transcribe_audio(asr_pipe, audio: np.ndarray, sample_rate: int) -> str:
    result = asr_pipe(
        {
            "array": np.asarray(audio, dtype=np.float32),
            "sampling_rate": sample_rate,
        }
    )
    if not isinstance(result, dict) or not result.get("text", "").strip():
        raise RuntimeError("Whisper returned an empty transcript.")
    return result["text"].strip()


class WaveformCanvas(tk.Canvas):
    def __init__(self, master, on_selection_changed, **kwargs):
        super().__init__(
            master,
            background="#111827",
            highlightthickness=1,
            highlightbackground="#334155",
            cursor="crosshair",
            **kwargs,
        )
        self.audio: np.ndarray | None = None
        self.sample_rate = REFERENCE_SAMPLE_RATE
        self.selection: tuple[float, float] | None = None
        self._anchor: float | None = None
        self._on_selection_changed = on_selection_changed
        self.bind("<Configure>", lambda _event: self.redraw())
        self.bind("<Button-1>", self._start_selection)
        self.bind("<B1-Motion>", self._drag_selection)
        self.bind("<ButtonRelease-1>", self._finish_selection)

    @property
    def duration(self) -> float:
        if self.audio is None:
            return 0.0
        return len(self.audio) / self.sample_rate

    def set_audio(self, audio: np.ndarray | None, sample_rate: int) -> None:
        self.audio = audio
        self.sample_rate = sample_rate
        self.selection = None
        self._anchor = None
        self.redraw()
        self._on_selection_changed(self.selection)

    def clear_selection(self) -> None:
        self.selection = None
        self._anchor = None
        self.redraw()
        self._on_selection_changed(self.selection)

    def _seconds_at_x(self, x: float) -> float:
        width = max(1, self.winfo_width())
        return max(0.0, min(self.duration, (x / width) * self.duration))

    def _start_selection(self, event) -> None:
        if self.audio is None or not len(self.audio):
            return
        self._anchor = self._seconds_at_x(event.x)
        self.selection = (self._anchor, self._anchor)
        self.redraw()

    def _drag_selection(self, event) -> None:
        if self._anchor is None:
            return
        self.selection = (self._anchor, self._seconds_at_x(event.x))
        self.redraw()
        self._on_selection_changed(self.selection)

    def _finish_selection(self, event) -> None:
        if self._anchor is None:
            return
        end = self._seconds_at_x(event.x)
        if abs(end - self._anchor) < 0.01:
            self.selection = None
        else:
            self.selection = (self._anchor, end)
        self._anchor = None
        self.redraw()
        self._on_selection_changed(self.selection)

    def redraw(self) -> None:
        self.delete("all")
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        center = height / 2

        self.create_line(0, center, width, center, fill="#334155")
        if self.audio is None or not len(self.audio):
            self.create_text(
                width / 2,
                center,
                text="Record or open a WAV file to begin",
                fill="#94a3b8",
                font=("Segoe UI", 11),
            )
            return

        if self.selection is not None:
            start, end = sorted(self.selection)
            start_x = (start / self.duration) * width
            end_x = (end / self.duration) * width
            self.create_rectangle(
                start_x,
                0,
                end_x,
                height,
                fill="#1d4ed8",
                stipple="gray50",
                outline="",
            )
            self.create_line(start_x, 0, start_x, height, fill="#93c5fd", width=2)
            self.create_line(end_x, 0, end_x, height, fill="#93c5fd", width=2)

        samples_per_column = max(1, int(np.ceil(len(self.audio) / width)))
        amplitude = max(1.0, float(np.max(np.abs(self.audio))))
        scale = (height * 0.39) / amplitude
        points = []
        for x in range(width):
            start = x * samples_per_column
            if start >= len(self.audio):
                break
            chunk = self.audio[start : start + samples_per_column]
            points.extend((x, center - float(np.max(chunk)) * scale))
            points.extend((x, center - float(np.min(chunk)) * scale))
        if points:
            self.create_line(*points, fill="#60a5fa", width=1)

        tick_count = max(2, min(10, round(self.duration)))
        for tick in range(tick_count + 1):
            seconds = (tick / tick_count) * self.duration
            x = (tick / tick_count) * width
            self.create_line(x, height - 14, x, height, fill="#475569")
            self.create_text(
                x + 4,
                height - 16,
                anchor="sw",
                text=f"{seconds:.1f}s",
                fill="#94a3b8",
                font=("Segoe UI", 8),
            )


class OmniVoiceStudio:
    def __init__(
        self,
        root: tk.Tk,
        model_path: str,
        device: str,
        asr_model: str,
        asr_device: str,
    ):
        self.root = root
        self.model_path = model_path
        self.inference_device = device
        self.asr_model_name = asr_model
        self.asr_device = asr_device
        self.sc = _load_soundcard()
        self.speakers = []
        self.mixer_microphones = []
        self.virtual_outputs = []
        self.audio: np.ndarray | None = None
        self.generated_audio: np.ndarray | None = None
        self.undo_stack: list[np.ndarray] = []
        self.events: queue.Queue = queue.Queue()
        self.record_stop = threading.Event()
        self.play_stop = threading.Event()
        self.record_started_at: float | None = None
        self.model = None
        self.asr_pipe = None
        self.busy_operation: str | None = None
        self._temp_reference: str | None = None
        self._transcription_after_id: str | None = None
        self._transcription_request_id = 0
        self._suppress_auto_transcription = False
        self.hotkey_manager = GlobalHotkeyManager(self.events)
        self.mixer = RealtimeMicMixer(self.events)
        self.live_voice_process: subprocess.Popen | None = None
        self.live_voice_active_preset = ""
        self.live_voice_infer_ms = 0.0
        (
            self.initial_record_hotkey,
            self.initial_stop_hotkey,
            self.initial_play_hotkey,
            self.initial_live_quality,
            self.hotkey_config_warning,
        ) = self._load_hotkey_settings()
        self.hotkey_specs: dict[str, HotkeySpec] = {}

        self._configure_window()
        self._build_ui()
        self._refresh_devices()
        self._apply_hotkeys(save=False)
        self._restore_live_reference()
        if self.hotkey_config_warning:
            self._set_status(self.hotkey_config_warning, "error")
        self._update_controls()
        self.root.after(50, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _restore_live_reference(self) -> None:
        if not SEED_VC_REFERENCE.exists():
            return
        try:
            data, sample_rate = sf.read(
                SEED_VC_REFERENCE,
                dtype="float32",
                always_2d=True,
            )
            audio = _prepare_audio(data, sample_rate, REFERENCE_SAMPLE_RATE)
            _validate_audio(audio)
            self._suppress_auto_transcription = True
            try:
                self._set_audio(audio)
            finally:
                self._suppress_auto_transcription = False
            self._set_status("Restored the previous live-clone reference.", "success")
        except (OSError, RuntimeError, ValueError) as exc:
            self._set_status(
                f"Could not restore the previous live reference: {exc}",
                "error",
            )

    def _configure_window(self) -> None:
        self.root.title("OmniVoice Studio")
        self.root.geometry("1420x900")
        self.root.minsize(1180, 760)
        self.root.configure(background="#f3f4f6")

        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 20))
        style.configure(
            "Subtitle.TLabel",
            foreground="#475569",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Record.TButton",
            foreground="#991b1b",
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "Primary.TButton",
            foreground="#1d4ed8",
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "Nav.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(18, 9),
        )
        style.configure(
            "NavActive.TButton",
            foreground="#1d4ed8",
            font=("Segoe UI Semibold", 10),
            padding=(18, 9),
        )
        style.configure(
            "Route.TLabel",
            foreground="#1d4ed8",
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "Section.TLabelframe.Label",
            font=("Segoe UI Semibold", 11),
        )
        style.configure("Status.TLabel", foreground="#475569")
        style.configure("Error.Status.TLabel", foreground="#b91c1c")
        style.configure("Success.Status.TLabel", foreground="#166534")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=(20, 16, 20, 12))
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="OmniVoice Studio", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Reference workspace, text clone, and live CS2 voice in one view.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        route_box = ttk.Frame(header)
        route_box.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Label(route_box, text="CS2 microphone", style="Subtitle.TLabel").grid(
            row=0, column=0, sticky="e"
        )
        ttk.Label(
            route_box,
            text="Voicemeeter Out 6 (B1)",
            style="Route.TLabel",
        ).grid(row=1, column=0, sticky="e")

        workspace = ttk.Panedwindow(outer, orient="horizontal")
        workspace.grid(row=1, column=0, sticky="nsew")

        reference_column = ttk.Frame(workspace, padding=(0, 0, 8, 0))
        reference_column.columnconfigure(0, weight=1)
        reference_column.rowconfigure(1, weight=1)
        workspace.add(reference_column, weight=5)

        capture_panel = ttk.Labelframe(
            reference_column,
            text="1. Capture and shape the reference",
            style="Section.TLabelframe",
            padding=10,
        )
        capture_panel.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        capture_panel.columnconfigure(1, weight=1)
        ttk.Label(capture_panel, text="System output").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.device_combo = ttk.Combobox(capture_panel, state="readonly")
        self.device_combo.grid(row=0, column=1, sticky="ew")
        ttk.Button(capture_panel, text="Refresh", command=self._refresh_devices).grid(
            row=0, column=2, padx=(6, 0)
        )
        transport = ttk.Frame(capture_panel)
        transport.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.record_button = ttk.Button(
            transport,
            text="Record",
            style="Record.TButton",
            command=self._start_recording,
        )
        self.record_button.pack(side="left")
        self.stop_button = ttk.Button(transport, text="Stop", command=self._stop_active)
        self.stop_button.pack(side="left", padx=(6, 0))
        self.play_button = ttk.Button(
            transport, text="Play selection", command=self._play_reference
        )
        self.play_button.pack(side="left", padx=(6, 0))
        self.open_button = ttk.Button(
            transport, text="Open WAV", command=self._open_reference
        )
        self.open_button.pack(side="right")
        self.save_button = ttk.Button(
            transport, text="Save", command=self._save_reference
        )
        self.save_button.pack(side="right", padx=(0, 6))

        waveform_panel = ttk.Labelframe(
            reference_column,
            text="Waveform",
            style="Section.TLabelframe",
            padding=10,
        )
        waveform_panel.grid(row=1, column=0, sticky="nsew")
        waveform_panel.columnconfigure(0, weight=1)
        waveform_panel.rowconfigure(0, weight=1)
        self.waveform = WaveformCanvas(
            waveform_panel,
            on_selection_changed=self._selection_changed,
            height=500,
        )
        self.waveform.grid(row=0, column=0, sticky="nsew")
        edit_row = ttk.Frame(waveform_panel)
        edit_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.selection_var = tk.StringVar(value="No selection")
        ttk.Label(edit_row, textvariable=self.selection_var).pack(side="left")
        self.clear_selection_button = ttk.Button(
            edit_row, text="Clear", command=self.waveform.clear_selection
        )
        self.clear_selection_button.pack(side="right")
        self.undo_button = ttk.Button(edit_row, text="Undo", command=self._undo)
        self.undo_button.pack(side="right", padx=(0, 6))
        self.delete_button = ttk.Button(
            edit_row, text="Delete", command=self._delete_selection
        )
        self.delete_button.pack(side="right", padx=(0, 6))
        self.keep_button = ttk.Button(
            edit_row, text="Keep selection", command=self._keep_selection
        )
        self.keep_button.pack(side="right", padx=(0, 6))

        clone_column = ttk.Frame(workspace, padding=(8, 0))
        clone_column.columnconfigure(0, weight=1)
        clone_column.rowconfigure(0, weight=1)
        clone_column.rowconfigure(1, weight=1)
        workspace.add(clone_column, weight=4)

        transcript_panel = ttk.Labelframe(
            clone_column,
            text="2. Review the reference",
            style="Section.TLabelframe",
            padding=10,
        )
        transcript_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        transcript_panel.columnconfigure(0, weight=1)
        transcript_panel.rowconfigure(2, weight=1)
        ttk.Label(
            transcript_panel,
            text="Whisper fills this automatically. Correct names or mistakes for text cloning.",
            style="Subtitle.TLabel",
            wraplength=390,
        ).grid(row=0, column=0, sticky="w")
        self.auto_transcribe_var = tk.BooleanVar(value=True)
        self.auto_transcribe_check = ttk.Checkbutton(
            transcript_panel,
            text="Auto-transcribe",
            variable=self.auto_transcribe_var,
            command=self._auto_transcribe_toggled,
        )
        self.auto_transcribe_check.grid(row=1, column=0, sticky="w", pady=(7, 5))
        self.transcript_text = tk.Text(
            transcript_panel,
            height=7,
            wrap="word",
            font=("Segoe UI", 10),
            relief="solid",
            borderwidth=1,
            undo=True,
        )
        self.transcript_text.grid(row=2, column=0, sticky="nsew")
        transcript_footer = ttk.Frame(transcript_panel)
        transcript_footer.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(
            transcript_footer,
            text="Best: 6-15 s, one clean speaker.",
            style="Subtitle.TLabel",
        ).pack(side="left")
        self.transcribe_button = ttk.Button(
            transcript_footer, text="Transcribe", command=self._transcribe_now
        )
        self.transcribe_button.pack(side="right")

        generate_panel = ttk.Labelframe(
            clone_column,
            text="3. Generate a text clip",
            style="Section.TLabelframe",
            padding=10,
        )
        generate_panel.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        generate_panel.columnconfigure(0, weight=1)
        generate_panel.rowconfigure(0, weight=1)
        self.clone_text = tk.Text(
            generate_panel,
            height=8,
            wrap="word",
            font=("Segoe UI", 10),
            relief="solid",
            borderwidth=1,
            undo=True,
        )
        self.clone_text.insert(
            "1.0", "This is a test of the voice captured in OmniVoice Studio."
        )
        self.clone_text.grid(row=0, column=0, columnspan=4, sticky="nsew")
        self.language_var = tk.StringVar(value="en")
        self.steps_var = tk.IntVar(value=16)
        ttk.Label(generate_panel, text="Language").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(generate_panel, textvariable=self.language_var, width=7).grid(
            row=2, column=0, sticky="w"
        )
        ttk.Label(generate_panel, text="Steps").grid(
            row=1, column=1, sticky="w", padx=(10, 0), pady=(8, 0)
        )
        ttk.Spinbox(
            generate_panel,
            from_=4,
            to=64,
            increment=4,
            textvariable=self.steps_var,
            width=6,
        ).grid(row=2, column=1, sticky="w", padx=(10, 0))
        self.generate_button = ttk.Button(
            generate_panel,
            text="Generate",
            style="Primary.TButton",
            command=self._generate_clone,
        )
        self.generate_button.grid(row=2, column=2, sticky="e", padx=(12, 0))
        self.play_generated_button = ttk.Button(
            generate_panel, text="Play", command=self._play_generated
        )
        self.play_generated_button.grid(row=2, column=3, padx=(6, 0))
        self.save_generated_button = ttk.Button(
            generate_panel, text="Save", command=self._save_generated
        )
        self.save_generated_button.grid(row=3, column=3, sticky="e", pady=(6, 0))
        self.send_to_mic_button = ttk.Button(
            generate_panel,
            text="Send latest clip to CS2 mixer",
            command=self._queue_generated_to_mixer,
        )
        self.send_to_mic_button.grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0)
        )

        live_column = ttk.Frame(workspace, padding=(8, 0, 0, 0))
        live_column.columnconfigure(0, weight=1)
        live_column.rowconfigure(1, weight=1)
        workspace.add(live_column, weight=4)

        live_panel = ttk.Labelframe(
            live_column,
            text="4. Go live in CS2",
            style="Section.TLabelframe",
            padding=12,
        )
        live_panel.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        live_panel.columnconfigure(1, weight=1)
        ttk.Label(live_panel, text="Reference").grid(row=0, column=0, sticky="w")
        self.live_reference_var = tk.StringVar(value="No reference loaded")
        ttk.Label(
            live_panel, textvariable=self.live_reference_var, style="Route.TLabel"
        ).grid(row=0, column=1, columnspan=2, sticky="w", padx=(8, 0))
        ttk.Label(live_panel, text="Quality").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.live_voice_quality_var = tk.StringVar(value=self.initial_live_quality)
        self.live_voice_quality_combo = ttk.Combobox(
            live_panel,
            state="readonly",
            textvariable=self.live_voice_quality_var,
            values=list(LIVE_VOICE_PRESETS),
            width=17,
        )
        self.live_voice_quality_combo.grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0)
        )
        self.live_voice_quality_combo.bind(
            "<<ComboboxSelected>>", self._live_quality_selected
        )
        self.start_live_voice_button = ttk.Button(
            live_panel,
            text="Start live clone",
            style="Primary.TButton",
            command=self._start_live_voice,
        )
        self.start_live_voice_button.grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0)
        )
        self.stop_live_voice_button = ttk.Button(
            live_panel, text="Stop live clone", command=self._stop_live_voice
        )
        self.stop_live_voice_button.grid(row=2, column=2, padx=(6, 0), pady=(12, 0))
        self.live_voice_status_var = tk.StringVar(
            value="Choose quality, then start live conversion."
        )
        ttk.Label(
            live_panel,
            textvariable=self.live_voice_status_var,
            style="Subtitle.TLabel",
            wraplength=380,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))

        mixer_panel = ttk.Labelframe(
            live_column,
            text="Game route and levels",
            style="Section.TLabelframe",
            padding=12,
        )
        mixer_panel.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        mixer_panel.columnconfigure(0, weight=1)
        ttk.Label(mixer_panel, text="Live microphone").grid(row=0, column=0, sticky="w")
        self.mixer_mic_combo = ttk.Combobox(mixer_panel, state="readonly")
        self.mixer_mic_combo.grid(row=1, column=0, sticky="ew", pady=(3, 8))
        ttk.Label(mixer_panel, text="Virtual feed").grid(row=2, column=0, sticky="w")
        self.virtual_output_combo = ttk.Combobox(mixer_panel, state="readonly")
        self.virtual_output_combo.grid(row=3, column=0, sticky="ew", pady=(3, 10))

        self.microphone_gain_var = tk.DoubleVar(value=1.0)
        self.generated_gain_var = tk.DoubleVar(value=1.0)
        self.microphone_gain_label = tk.StringVar(value="Mic 100%")
        self.generated_gain_label = tk.StringVar(value="AI 100%")
        ttk.Label(mixer_panel, textvariable=self.microphone_gain_label).grid(
            row=4, column=0, sticky="w"
        )
        self.microphone_gain_scale = ttk.Scale(
            mixer_panel,
            from_=0.0,
            to=2.0,
            variable=self.microphone_gain_var,
            command=lambda _value: self._mixer_gain_changed(),
        )
        self.microphone_gain_scale.grid(row=5, column=0, sticky="ew")
        ttk.Label(mixer_panel, textvariable=self.generated_gain_label).grid(
            row=6, column=0, sticky="w", pady=(8, 0)
        )
        self.generated_gain_scale = ttk.Scale(
            mixer_panel,
            from_=0.0,
            to=2.0,
            variable=self.generated_gain_var,
            command=lambda _value: self._mixer_gain_changed(),
        )
        self.generated_gain_scale.grid(row=7, column=0, sticky="ew")

        meter_row = ttk.Frame(mixer_panel)
        meter_row.grid(row=8, column=0, sticky="ew", pady=(12, 0))
        meter_row.columnconfigure(1, weight=1)
        ttk.Label(meter_row, text="Mic").grid(row=0, column=0, padx=(0, 6))
        self.microphone_meter = ttk.Progressbar(meter_row, maximum=100)
        self.microphone_meter.grid(row=0, column=1, sticky="ew")
        ttk.Label(meter_row, text="AI").grid(row=1, column=0, padx=(0, 6), pady=(5, 0))
        self.generated_meter = ttk.Progressbar(meter_row, maximum=100)
        self.generated_meter.grid(row=1, column=1, sticky="ew", pady=(5, 0))
        ttk.Label(meter_row, text="Out").grid(row=2, column=0, padx=(0, 6), pady=(5, 0))
        self.output_meter = ttk.Progressbar(meter_row, maximum=100)
        self.output_meter.grid(row=2, column=1, sticky="ew", pady=(5, 0))

        mixer_actions = ttk.Frame(mixer_panel)
        mixer_actions.grid(row=9, column=0, sticky="ew", pady=(12, 0))
        self.start_mixer_button = ttk.Button(
            mixer_actions, text="Start normal mixer", command=self._start_mixer
        )
        self.start_mixer_button.pack(side="left")
        self.stop_mixer_button = ttk.Button(
            mixer_actions, text="Stop", command=self._stop_mixer
        )
        self.stop_mixer_button.pack(side="left", padx=(6, 0))
        self.mixer_status_var = tk.StringVar(value="CS2 input: Voicemeeter Out 6")
        ttk.Label(
            mixer_panel,
            textvariable=self.mixer_status_var,
            style="Subtitle.TLabel",
            wraplength=360,
        ).grid(row=10, column=0, sticky="w", pady=(10, 0))
        self.live_ready_var = tk.StringVar(value="Reference: missing")
        ttk.Label(
            mixer_panel, textvariable=self.live_ready_var, style="Route.TLabel"
        ).grid(row=11, column=0, sticky="w", pady=(12, 0))

        hotkey_panel = ttk.Labelframe(
            outer,
            text="Global hotkeys - active while CS2 is focused",
            style="Section.TLabelframe",
            padding=(10, 7),
        )
        hotkey_panel.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        hotkey_panel.columnconfigure(1, weight=1)
        hotkey_panel.columnconfigure(4, weight=1)
        ttk.Label(hotkey_panel, text="Record").grid(row=0, column=0, sticky="w")
        self.record_hotkey_var = tk.StringVar(value=self.initial_record_hotkey)
        self.record_hotkey_entry = ttk.Entry(
            hotkey_panel, textvariable=self.record_hotkey_var
        )
        self.record_hotkey_entry.grid(row=0, column=1, sticky="ew", padx=(6, 4))
        self.detect_record_hotkey_button = ttk.Button(
            hotkey_panel, text="Detect", command=lambda: self._detect_hotkey("record")
        )
        self.detect_record_hotkey_button.grid(row=0, column=2)
        ttk.Label(hotkey_panel, text="Stop").grid(row=0, column=3, padx=(14, 0))
        self.stop_hotkey_var = tk.StringVar(value=self.initial_stop_hotkey)
        self.stop_hotkey_entry = ttk.Entry(
            hotkey_panel, textvariable=self.stop_hotkey_var
        )
        self.stop_hotkey_entry.grid(row=0, column=4, sticky="ew", padx=(6, 4))
        self.detect_stop_hotkey_button = ttk.Button(
            hotkey_panel, text="Detect", command=lambda: self._detect_hotkey("stop")
        )
        self.detect_stop_hotkey_button.grid(row=0, column=5)
        ttk.Label(hotkey_panel, text="Play AI").grid(row=0, column=6, padx=(14, 0))
        self.play_hotkey_var = tk.StringVar(value=self.initial_play_hotkey)
        self.play_hotkey_entry = ttk.Entry(
            hotkey_panel, textvariable=self.play_hotkey_var, width=15
        )
        self.play_hotkey_entry.grid(row=0, column=7, padx=(6, 4))
        self.detect_play_hotkey_button = ttk.Button(
            hotkey_panel, text="Detect", command=lambda: self._detect_hotkey("play")
        )
        self.detect_play_hotkey_button.grid(row=0, column=8)
        self.apply_hotkeys_button = ttk.Button(
            hotkey_panel, text="Apply", command=self._apply_hotkeys
        )
        self.apply_hotkeys_button.grid(row=0, column=9, padx=(10, 0))
        self.hotkey_status_var = tk.StringVar(value="Not registered")
        ttk.Label(
            hotkey_panel, textvariable=self.hotkey_status_var, style="Subtitle.TLabel"
        ).grid(row=1, column=0, columnspan=10, sticky="w", pady=(5, 0))

        footer = ttk.Frame(outer)
        footer.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        footer.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            footer, textvariable=self.status_var, style="Status.TLabel"
        )
        self.status_label.grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=180)
        self.progress.grid(row=0, column=1, sticky="e")
        self.progress.grid_remove()

        self.root.bind("<Control-o>", lambda _event: self._open_reference())
        self.root.bind("<Control-s>", lambda _event: self._save_reference())
        self.root.bind("<Control-z>", lambda _event: self._undo())
        self.root.bind("<Escape>", lambda _event: self._stop_active())

    def _load_hotkey_settings(
        self,
    ) -> tuple[str, str, str, str, str | None]:
        if not HOTKEY_CONFIG_PATH.exists():
            return (
                DEFAULT_RECORD_HOTKEY,
                DEFAULT_STOP_HOTKEY,
                DEFAULT_PLAY_HOTKEY,
                "High fidelity",
                None,
            )
        try:
            settings = json.loads(HOTKEY_CONFIG_PATH.read_text(encoding="utf-8"))
            record = parse_hotkey(str(settings["record_hotkey"])).text
            stop = parse_hotkey(str(settings["stop_hotkey"])).text
            play = parse_hotkey(
                str(settings.get("play_hotkey", DEFAULT_PLAY_HOTKEY))
            ).text
            quality = str(settings.get("live_quality", "High fidelity"))
            if quality not in LIVE_VOICE_PRESETS:
                raise ValueError(f"Unknown live quality preset: {quality}")
            if len({record, stop, play}) != 3:
                raise ValueError("Record, stop, and play hotkeys must differ.")
            return record, stop, play, quality, None
        except (
            json.JSONDecodeError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            return (
                DEFAULT_RECORD_HOTKEY,
                DEFAULT_STOP_HOTKEY,
                DEFAULT_PLAY_HOTKEY,
                "High fidelity",
                f"Hotkey settings were invalid; defaults restored: {exc}",
            )

    def _save_hotkey_settings(self) -> None:
        HOTKEY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        HOTKEY_CONFIG_PATH.write_text(
            json.dumps(
                {
                    "record_hotkey": self.record_hotkey_var.get(),
                    "stop_hotkey": self.stop_hotkey_var.get(),
                    "play_hotkey": self.play_hotkey_var.get(),
                    "live_quality": self.live_voice_quality_var.get(),
                },
                indent=2,
            )
            + os.linesep,
            encoding="utf-8",
        )

    def _live_quality_selected(self, _event=None) -> None:
        try:
            self._save_hotkey_settings()
            self._set_status(
                f"Live quality set to {self.live_voice_quality_var.get()}.",
                "success",
            )
        except OSError as exc:
            self._set_status(
                f"Could not save live quality: {exc}",
                "error",
            )

    def _restore_active_hotkeys(self) -> None:
        if not self.hotkey_specs:
            return
        try:
            self.hotkey_manager.start(self.hotkey_specs)
            record = self.hotkey_specs["record"].text
            stop = self.hotkey_specs["stop"].text
            play = self.hotkey_specs["play"].text
            self.hotkey_status_var.set(
                f"Active: {record} starts, {stop} stops, {play} plays AI"
            )
        except RuntimeError as exc:
            self.hotkey_status_var.set("Hotkeys are not active")
            self._set_status(f"Could not restore hotkeys: {exc}", "error")

    def _detect_hotkey(self, action: str) -> None:
        if self.busy_operation is not None:
            return
        self.hotkey_manager.stop()
        self.hotkey_status_var.set("Paused while detecting a new hotkey")

        action_label = {
            "record": "record",
            "stop": "stop",
            "play": "play AI",
        }[action]
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Detect {action_label} hotkey")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.configure(background="#f3f4f6")
        dialog_width = 430
        dialog_height = 180
        dialog_x = (
            self.root.winfo_rootx() + (self.root.winfo_width() - dialog_width) // 2
        )
        dialog_y = (
            self.root.winfo_rooty() + (self.root.winfo_height() - dialog_height) // 2
        )
        dialog.geometry(f"{dialog_width}x{dialog_height}+{dialog_x}+{dialog_y}")
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=22)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text=f"Press the new {action_label} hotkey",
            font=("Segoe UI Semibold", 14),
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="Hold any modifiers, then press the main key.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 14))
        detected_var = tk.StringVar(value="Waiting for keys...")
        ttk.Label(
            frame,
            textvariable=detected_var,
            font=("Segoe UI Semibold", 12),
        ).pack(anchor="w")
        ttk.Button(frame, text="Cancel", command=lambda: finish(None)).pack(
            anchor="e", pady=(18, 0)
        )

        pressed_modifiers: set[str] = set()
        finished = False

        def finish(spec: HotkeySpec | None) -> None:
            nonlocal finished
            if finished:
                return
            finished = True
            dialog.grab_release()
            dialog.destroy()
            if spec is None:
                self._restore_active_hotkeys()
                self._set_status("Hotkey detection cancelled.")
                return

            variables = {
                "record": self.record_hotkey_var,
                "stop": self.stop_hotkey_var,
                "play": self.play_hotkey_var,
            }
            for existing_action, existing_spec in self.hotkey_specs.items():
                variables[existing_action].set(existing_spec.text)
            variables[action].set(spec.text)
            self._apply_hotkeys()

        def update_preview() -> None:
            ordered = [
                modifier
                for modifier in ("Ctrl", "Alt", "Shift", "Win")
                if modifier in pressed_modifiers
            ]
            detected_var.set(
                "+".join(ordered) + ("+..." if ordered else "Waiting for keys...")
            )

        def on_key_press(event):
            if event.keysym == "Escape":
                finish(None)
                return "break"
            modifier = _TK_MODIFIER_KEYSYMS.get(event.keysym)
            if modifier is not None:
                pressed_modifiers.add(modifier)
                update_preview()
                return "break"

            event_modifiers = set(pressed_modifiers)
            if event.state & 0x0001:
                event_modifiers.add("Shift")
            if event.state & 0x0004:
                event_modifiers.add("Ctrl")
            if event.state & 0x0008:
                event_modifiers.add("Alt")
            try:
                spec = detected_hotkey(event_modifiers, event.keysym)
            except ValueError as exc:
                detected_var.set(str(exc))
                return "break"
            detected_var.set(spec.text)
            dialog.after(250, lambda: finish(spec))
            return "break"

        def on_key_release(event):
            modifier = _TK_MODIFIER_KEYSYMS.get(event.keysym)
            if modifier is not None:
                pressed_modifiers.discard(modifier)
                update_preview()
            return "break"

        dialog.bind("<KeyPress>", on_key_press)
        dialog.bind("<KeyRelease>", on_key_release)
        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(None))
        dialog.after(50, dialog.focus_force)

    def _apply_hotkeys(self, save: bool = True) -> None:
        try:
            record_spec = parse_hotkey(self.record_hotkey_var.get())
            stop_spec = parse_hotkey(self.stop_hotkey_var.get())
            play_spec = parse_hotkey(self.play_hotkey_var.get())
            combinations = {
                (record_spec.modifiers, record_spec.virtual_key),
                (stop_spec.modifiers, stop_spec.virtual_key),
                (play_spec.modifiers, play_spec.virtual_key),
            }
            if len(combinations) != 3:
                raise ValueError("Record, stop, and play hotkeys must be different.")
        except ValueError as exc:
            if self.hotkey_specs:
                self._restore_active_hotkeys()
            else:
                self.hotkey_status_var.set("Hotkeys are not active")
            self._set_status(f"Could not apply hotkeys: {exc}", "error")
            return

        previous_specs = self.hotkey_specs.copy()
        self.hotkey_manager.stop()
        try:
            self.hotkey_manager.start(
                {
                    "record": record_spec,
                    "stop": stop_spec,
                    "play": play_spec,
                }
            )
        except RuntimeError as exc:
            if previous_specs:
                try:
                    self.hotkey_manager.start(previous_specs)
                    record_previous = previous_specs["record"].text
                    stop_previous = previous_specs["stop"].text
                    play_previous = previous_specs["play"].text
                    self.hotkey_status_var.set(
                        f"Active: {record_previous} starts, "
                        f"{stop_previous} stops, {play_previous} plays AI"
                    )
                    self._set_status(
                        f"Could not apply new hotkeys; previous bindings remain: {exc}",
                        "error",
                    )
                    return
                except RuntimeError as restore_exc:
                    self._set_status(
                        f"Could not register hotkeys: {exc}; restoring the "
                        f"previous bindings also failed: {restore_exc}",
                        "error",
                    )
            else:
                self._set_status(f"Could not register hotkeys: {exc}", "error")
            self.hotkey_status_var.set("Hotkeys are not active")
            return

        self.hotkey_specs = {
            "record": record_spec,
            "stop": stop_spec,
            "play": play_spec,
        }
        self.record_hotkey_var.set(record_spec.text)
        self.stop_hotkey_var.set(stop_spec.text)
        self.play_hotkey_var.set(play_spec.text)
        self.hotkey_status_var.set(
            f"Active: {record_spec.text} starts, {stop_spec.text} stops, "
            f"{play_spec.text} plays AI"
        )
        if save:
            try:
                self._save_hotkey_settings()
            except OSError as exc:
                self._set_status(
                    f"Hotkeys are active for this session but could not be saved: {exc}",
                    "error",
                )
                return
        self._set_status("Global recording hotkeys are active.", "success")

    def _refresh_devices(self) -> None:
        try:
            default = self.sc.default_speaker()
            self.speakers = list(self.sc.all_speakers())
            values = []
            default_index = 0
            for index, speaker in enumerate(self.speakers):
                marker = " (default)" if default and speaker.id == default.id else ""
                values.append(f"{index}: {speaker.name}{marker}")
                if marker and not is_virtual_audio_device_name(speaker.name):
                    default_index = index
            if default is None or is_virtual_audio_device_name(default.name):
                physical_speakers = [
                    (index, speaker)
                    for index, speaker in enumerate(self.speakers)
                    if not is_virtual_audio_device_name(speaker.name)
                ]
                if physical_speakers:
                    default_index = next(
                        (
                            index
                            for index, speaker in physical_speakers
                            if "headphone" in speaker.name.casefold()
                        ),
                        physical_speakers[0][0],
                    )
            self.device_combo.configure(values=values)
            if values:
                self.device_combo.current(default_index)
                self._set_status(
                    f"Ready to capture {self.speakers[default_index].name}"
                )
            else:
                self._set_status("No system output devices were found.", "error")

            import sounddevice as sd

            try:
                host_apis = sd.query_hostapis()
                wasapi_index = next(
                    index
                    for index, host_api in enumerate(host_apis)
                    if host_api["name"] == "Windows WASAPI"
                )
                devices = sd.query_devices()
            except (StopIteration, sd.PortAudioError) as exc:
                raise RuntimeError(
                    f"Could not enumerate Windows audio mixer devices: {exc}"
                ) from exc

            default_microphone_index = host_apis[wasapi_index]["default_input_device"]
            self.mixer_microphones = [
                PortAudioDevice(index, device["name"])
                for index, device in enumerate(devices)
                if device["hostapi"] == wasapi_index
                and device["max_input_channels"] > 0
                and is_physical_microphone_name(device["name"])
            ]
            microphone_values = [
                microphone.name for microphone in self.mixer_microphones
            ]
            self.mixer_mic_combo.configure(values=microphone_values)
            microphone_index = 0
            for index, microphone in enumerate(self.mixer_microphones):
                if microphone.index == default_microphone_index:
                    microphone_index = index
                    break
            if microphone_values:
                self.mixer_mic_combo.current(microphone_index)

            self.virtual_outputs = [
                PortAudioDevice(index, device["name"])
                for index, device in enumerate(devices)
                if device["hostapi"] == wasapi_index
                and device["max_output_channels"] >= 2
                and is_virtual_output_name(device["name"])
            ]
            virtual_values = [speaker.name for speaker in self.virtual_outputs]
            self.virtual_output_combo.configure(values=virtual_values)
            virtual_index = 0
            for index, speaker in enumerate(self.virtual_outputs):
                if speaker.name.casefold().startswith("voicemeeter input"):
                    virtual_index = index
                    break
            if virtual_values:
                self.virtual_output_combo.current(virtual_index)
                self.mixer_status_var.set(
                    "Set CS2 microphone input to Voicemeeter Out 6 (B1)."
                )
            else:
                self.mixer_status_var.set(
                    "No virtual output found. Install VoiceMeeter or VB-CABLE."
                )
        except (RuntimeError, OSError) as exc:
            self._set_status(f"Could not list audio devices: {exc}", "error")
        self._update_controls()

    def _selected_speaker(self):
        index = self.device_combo.current()
        if index < 0 or index >= len(self.speakers):
            raise RuntimeError("Select a system output device first.")
        return self.speakers[index]

    def _selected_mixer_microphone(self):
        index = self.mixer_mic_combo.current()
        if index < 0 or index >= len(self.mixer_microphones):
            raise RuntimeError("Select a live microphone first.")
        return self.mixer_microphones[index]

    def _selected_virtual_output(self):
        index = self.virtual_output_combo.current()
        if index < 0 or index >= len(self.virtual_outputs):
            raise RuntimeError("Select a virtual output such as VoiceMeeter Input.")
        return self.virtual_outputs[index]

    def _mixer_gain_changed(self) -> None:
        microphone_gain = float(self.microphone_gain_var.get())
        generated_gain = float(self.generated_gain_var.get())
        self.microphone_gain_label.set(f"Mic {round(microphone_gain * 100)}%")
        self.generated_gain_label.set(f"AI {round(generated_gain * 100)}%")
        self.mixer.set_gains(microphone_gain, generated_gain)

    def _start_mixer(self) -> None:
        if self.mixer.running:
            return
        try:
            microphone = self._selected_mixer_microphone()
            virtual_output = self._selected_virtual_output()
            if virtual_output.name.casefold().startswith("voicemeeter input"):
                configure_voicemeeter_b1()
            self._mixer_gain_changed()
            self.mixer.start(microphone.index, virtual_output.index)
            self.mixer_status_var.set("Starting game microphone mixer...")
            self._set_status("Starting microphone and AI audio route to CS2...")
        except (OSError, RuntimeError, ValueError) as exc:
            self.mixer_status_var.set(f"Mixer could not start: {exc}")
            self._set_status(f"Mixer could not start: {exc}", "error")
        self._update_controls()

    def _stop_mixer(self) -> None:
        if not self.mixer.running:
            return
        self.mixer_status_var.set("Stopping game microphone mixer...")
        self.mixer.stop()
        self._update_controls()

    def _queue_generated_to_mixer(self) -> None:
        if self.generated_audio is None:
            self._set_status("Generate cloned speech first.", "error")
            return
        try:
            self.mixer.queue_generated(self.generated_audio)
            self.mixer_status_var.set(
                "Latest cloned speech queued for the CS2 microphone."
            )
            self._set_status("Queued cloned speech to the game microphone.", "success")
        except RuntimeError as exc:
            self._set_status(str(exc), "error")

    def _live_voice_running(self) -> bool:
        return (
            self.live_voice_process is not None
            and self.live_voice_process.poll() is None
        )

    def _prepare_live_voice_reference(self) -> tuple[Path, str | None]:
        audio = self._reference_slice()
        duration = len(audio) / REFERENCE_SAMPLE_RATE
        if not 1.0 <= duration <= 30.0:
            raise ValueError(
                "Live voice conversion needs a 1-30 second reference clip."
            )
        SEED_VC_REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            SEED_VC_REFERENCE,
            audio,
            REFERENCE_SAMPLE_RATE,
            subtype="PCM_16",
        )
        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
        peak = float(np.max(np.abs(audio), initial=0.0))
        warnings = []
        if duration < 6.0:
            warnings.append("use 6-15 seconds for closer similarity")
        elif duration > 15.0:
            warnings.append("trim to the cleanest 6-15 seconds")
        if rms < 0.02:
            warnings.append("the reference level is low")
        if peak >= 0.999:
            warnings.append("the reference may be clipped")
        warning = "; ".join(warnings) if warnings else None
        return SEED_VC_REFERENCE, warning

    def _release_gpu_models(self) -> None:
        self.model = None
        self.asr_pipe = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _start_live_voice(self) -> None:
        if self._live_voice_running():
            return
        if self.busy_operation is not None:
            self._set_status(
                "Wait for the current recording or generation task to finish.",
                "error",
            )
            return
        if not SEED_VC_PYTHON.exists() or not LIVE_VOICE_WORKER.exists():
            self._set_status(
                "Seed-VC live mode is not installed correctly.",
                "error",
            )
            return
        try:
            reference_path, reference_warning = self._prepare_live_voice_reference()
            microphone = self._selected_mixer_microphone()
            virtual_output = self._selected_virtual_output()
            if not virtual_output.name.casefold().startswith("voicemeeter input"):
                raise ValueError(
                    "Live AI mode requires Voicemeeter Input as the virtual feed."
                )
            configure_voicemeeter_b1()
            if self.mixer.running:
                self.mixer.stop()
            self._release_gpu_models()
            preset_name = self.live_voice_quality_var.get()
            preset = LIVE_VOICE_PRESETS[preset_name]
            self.live_voice_active_preset = preset_name

            command = [
                str(SEED_VC_PYTHON),
                str(LIVE_VOICE_WORKER),
                "--seed-vc-dir",
                str(SEED_VC_DIRECTORY),
                "--reference",
                str(reference_path),
                "--input-device",
                microphone.name,
                "--output-device",
                virtual_output.name,
                "--diffusion-steps",
                str(preset["diffusion_steps"]),
                "--inference-cfg-rate",
                str(preset["inference_cfg_rate"]),
                "--max-prompt-length",
                str(preset["max_prompt_length"]),
                "--block-time",
                str(preset["block_time"]),
                "--extra-time-ce",
                str(preset["extra_time_ce"]),
                "--extra-time",
                str(preset["extra_time"]),
            ]
            self.live_voice_process = subprocess.Popen(
                command,
                cwd=SEED_VC_DIRECTORY,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self.live_voice_process = None
            self.live_voice_status_var.set(f"Could not start: {exc}")
            self._set_status(f"Could not start live AI voice: {exc}", "error")
            self._update_controls()
            return

        loading_status = f"Loading Seed-VC ({self.live_voice_active_preset})..."
        if reference_warning:
            loading_status += f" Reference note: {reference_warning}."
        self.live_voice_status_var.set(loading_status)
        self._set_status("Starting zero-shot live voice conversion...")
        threading.Thread(
            target=self._watch_live_voice,
            args=(self.live_voice_process,),
            daemon=True,
            name="OmniVoiceLiveVoiceWatcher",
        ).start()
        self._update_controls()

    def _watch_live_voice(self, process: subprocess.Popen) -> None:
        log_file = None
        underrun_times = []
        overload_notified = False
        try:
            log_file = SEED_VC_LOG.open("a", encoding="utf-8")
        except OSError:
            pass
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    if log_file is not None:
                        log_file.write(line)
                        log_file.flush()
                    stripped = line.strip()
                    if stripped.startswith("READY"):
                        self.events.put(("live_voice_ready", stripped))
                    elif stripped.startswith("ERROR"):
                        self.events.put(("live_voice_error", stripped[6:]))
                    elif stripped.startswith("PERF infer_ms="):
                        self.events.put(("live_voice_perf", stripped.partition("=")[2]))
                    elif "output underrun" in stripped:
                        now = time.monotonic()
                        underrun_times = [
                            timestamp
                            for timestamp in underrun_times
                            if now - timestamp <= 5.0
                        ]
                        underrun_times.append(now)
                        if len(underrun_times) >= 3 and not overload_notified:
                            overload_notified = True
                            self.events.put(("live_voice_overloaded",))
            return_code = process.wait()
            self.events.put(("live_voice_stopped", return_code))
        finally:
            if log_file is not None:
                log_file.close()

    def _stop_live_voice(self) -> None:
        process = self.live_voice_process
        if process is None or process.poll() is not None:
            return
        self.live_voice_status_var.set("Stopping live voice conversion...")
        try:
            if process.stdin is not None:
                process.stdin.write("STOP\n")
                process.stdin.flush()
            else:
                process.terminate()
        except (BrokenPipeError, OSError):
            process.terminate()
        self._update_controls()

    def _force_stop_live_voice(self) -> None:
        process = self.live_voice_process
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write("STOP\n")
                process.stdin.flush()
            process.wait(timeout=10)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def _start_recording(self) -> None:
        if self._live_voice_running():
            self._set_status(
                "Stop live voice conversion before recording a new reference.",
                "error",
            )
            return
        if self.busy_operation is not None:
            return
        try:
            speaker = self._selected_speaker()
            loopback = self.sc.get_microphone(id=speaker.id, include_loopback=True)
            if loopback is None or not loopback.isloopback:
                raise RuntimeError(
                    f"No speaker-loopback recorder is available for {speaker.name}."
                )
        except (RuntimeError, OSError) as exc:
            self._set_status(str(exc), "error")
            return

        self.record_stop.clear()
        self.record_started_at = time.monotonic()
        self.busy_operation = "recording"
        self._set_status("Recording 0.0s - play only the target voice")
        self._update_controls()

        def worker():
            chunks = []
            try:
                with loopback.recorder(samplerate=CAPTURE_SAMPLE_RATE) as recorder:
                    while not self.record_stop.is_set():
                        chunks.append(recorder.record(numframes=PLAYBACK_BLOCK_SIZE))
                        elapsed = time.monotonic() - self.record_started_at
                        if elapsed >= MAX_RECORDING_SECONDS:
                            self.record_stop.set()
                if not chunks:
                    raise RuntimeError("No audio frames were captured.")
                recorded = np.concatenate(chunks, axis=0)
                audio = _prepare_audio(
                    recorded, CAPTURE_SAMPLE_RATE, REFERENCE_SAMPLE_RATE
                )
                peak, rms = _validate_audio(audio)
                self.events.put(("recorded", audio, peak, rms))
            except (RuntimeError, OSError, ValueError) as exc:
                self.events.put(("error", "Recording failed", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._update_recording_clock)

    def _update_recording_clock(self) -> None:
        if self.busy_operation != "recording" or self.record_started_at is None:
            return
        elapsed = time.monotonic() - self.record_started_at
        self._set_status(f"Recording {elapsed:.1f}s - play only the target voice")
        self.root.after(100, self._update_recording_clock)

    def _stop_active(self) -> None:
        if self.busy_operation == "recording":
            self.record_stop.set()
            self._set_status("Finishing recording...")
        elif self.busy_operation == "playback":
            self.play_stop.set()
            self._set_status("Stopping playback...")

    def _set_audio(self, audio: np.ndarray, push_undo: bool = False) -> None:
        if push_undo and self.audio is not None:
            self.undo_stack.append(self.audio.copy())
            self.undo_stack = self.undo_stack[-20:]
        self.audio = np.asarray(audio, dtype=np.float32)
        self.generated_audio = None
        self.waveform.set_audio(self.audio, REFERENCE_SAMPLE_RATE)
        self._update_controls()

    def _selection_changed(self, selection: tuple[float, float] | None) -> None:
        if self.audio is None:
            self.selection_var.set("No selection")
            self.live_reference_var.set("No reference loaded")
            self.live_ready_var.set("Reference: missing")
        elif selection is None:
            duration = len(self.audio) / REFERENCE_SAMPLE_RATE
            self.selection_var.set(f"Full clip: {duration:.2f}s")
            self.live_reference_var.set(f"Full edited clip - {duration:.2f}s")
            self.live_ready_var.set(f"Reference: ready ({duration:.2f}s)")
        else:
            start, end = sorted(selection)
            self.selection_var.set(
                f"Selection: {start:.2f}s - {end:.2f}s ({end - start:.2f}s)"
            )
            self.live_reference_var.set(f"Selected range - {end - start:.2f}s")
            self.live_ready_var.set(f"Reference: selected ({end - start:.2f}s)")
        self._update_controls()
        self._schedule_auto_transcription()

    def _auto_transcribe_toggled(self) -> None:
        if self.auto_transcribe_var.get():
            self._schedule_auto_transcription(delay_ms=0)
        else:
            self._transcription_request_id += 1
            if self._transcription_after_id is not None:
                self.root.after_cancel(self._transcription_after_id)
                self._transcription_after_id = None

    def _schedule_auto_transcription(self, delay_ms: int = 650) -> None:
        if (
            self._suppress_auto_transcription
            or not hasattr(self, "auto_transcribe_var")
            or not self.auto_transcribe_var.get()
            or self.audio is None
            or not len(self.audio)
        ):
            return
        self._transcription_request_id += 1
        request_id = self._transcription_request_id
        if self._transcription_after_id is not None:
            self.root.after_cancel(self._transcription_after_id)
        self._transcription_after_id = self.root.after(
            delay_ms,
            lambda: self._start_transcription(request_id),
        )

    def _transcribe_now(self) -> None:
        if self.audio is None:
            self._set_status("Record or open a reference clip first.", "error")
            return
        self._transcription_request_id += 1
        request_id = self._transcription_request_id
        if self._transcription_after_id is not None:
            self.root.after_cancel(self._transcription_after_id)
            self._transcription_after_id = None
        self._start_transcription(request_id)

    def _start_transcription(self, request_id: int) -> None:
        self._transcription_after_id = None
        if request_id != self._transcription_request_id:
            return
        if self.busy_operation is not None:
            self._transcription_after_id = self.root.after(
                400,
                lambda: self._start_transcription(request_id),
            )
            return
        try:
            audio = self._reference_slice()
        except (RuntimeError, ValueError) as exc:
            self._set_status(str(exc), "error")
            return

        self.busy_operation = "transcription"
        self._set_status(
            "Loading Whisper for automatic transcription..."
            if self.asr_pipe is None
            else "Transcribing reference audio..."
        )
        self.progress.grid()
        self.progress.start(12)
        self._update_controls()

        def worker():
            try:
                if self.asr_pipe is None:
                    import torch
                    from transformers import pipeline as hf_pipeline

                    dtype = (
                        torch.float16
                        if self.asr_device.startswith(("cuda", "xpu"))
                        else torch.float32
                    )
                    self.asr_pipe = hf_pipeline(
                        "automatic-speech-recognition",
                        model=self.asr_model_name,
                        dtype=dtype,
                        device=self.asr_device,
                    )
                transcript = transcribe_audio(
                    self.asr_pipe, audio, REFERENCE_SAMPLE_RATE
                )
                self.events.put(("transcribed", transcript, request_id))
            except (
                ImportError,
                MemoryError,
                OSError,
                RuntimeError,
                ValueError,
            ) as exc:
                self.events.put(("error", "Transcription failed", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _keep_selection(self) -> None:
        if self.audio is None:
            return
        try:
            result = keep_selection(
                self.audio, REFERENCE_SAMPLE_RATE, self.waveform.selection
            )
            self._set_audio(result, push_undo=True)
            self._set_status("Kept the selected audio.", "success")
        except ValueError as exc:
            self._set_status(str(exc), "error")

    def _delete_selection(self) -> None:
        if self.audio is None:
            return
        try:
            result = delete_selection(
                self.audio, REFERENCE_SAMPLE_RATE, self.waveform.selection
            )
            self._set_audio(result, push_undo=True)
            self._set_status("Deleted the selected audio.", "success")
        except ValueError as exc:
            self._set_status(str(exc), "error")

    def _undo(self) -> None:
        if not self.undo_stack or self.busy_operation is not None:
            return
        self.audio = self.undo_stack.pop()
        self.generated_audio = None
        self.waveform.set_audio(self.audio, REFERENCE_SAMPLE_RATE)
        self._set_status("Restored the previous edit.", "success")
        self._update_controls()

    def _reference_slice(self) -> np.ndarray:
        if self.audio is None:
            raise RuntimeError("Record or open a reference clip first.")
        return keep_selection(
            self.audio, REFERENCE_SAMPLE_RATE, self.waveform.selection
        )

    def _play_reference(self) -> None:
        try:
            self._start_playback(self._reference_slice(), "reference")
        except (RuntimeError, ValueError) as exc:
            self._set_status(str(exc), "error")

    def _play_generated(self) -> None:
        if self.generated_audio is None:
            self._set_status("Generate cloned speech first.", "error")
            return
        self._start_playback(self.generated_audio, "generated result")

    def _start_playback(self, audio: np.ndarray, label: str) -> None:
        if self.busy_operation is not None:
            return
        try:
            speaker = self._selected_speaker()
        except RuntimeError as exc:
            self._set_status(str(exc), "error")
            return

        self.play_stop.clear()
        self.busy_operation = "playback"
        self._set_status(f"Playing {label}...")
        self._update_controls()

        def worker():
            try:
                with speaker.player(
                    samplerate=REFERENCE_SAMPLE_RATE, channels=1
                ) as player:
                    for start in range(0, len(audio), PLAYBACK_BLOCK_SIZE):
                        if self.play_stop.is_set():
                            break
                        block = audio[start : start + PLAYBACK_BLOCK_SIZE]
                        player.play(block[:, np.newaxis])
                self.events.put(("playback_done",))
            except (RuntimeError, OSError, ValueError) as exc:
                self.events.put(("error", "Playback failed", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _open_reference(self) -> None:
        if self.busy_operation is not None:
            return
        path = filedialog.askopenfilename(
            title="Open reference audio",
            filetypes=[
                ("Audio files", "*.wav *.flac *.ogg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            audio = _prepare_audio(data, sample_rate, REFERENCE_SAMPLE_RATE)
            _validate_audio(audio)
            self.undo_stack.clear()
            transcript_path = Path(path).with_suffix(".txt")
            self._suppress_auto_transcription = transcript_path.exists()
            try:
                self._set_audio(audio)
                if transcript_path.exists():
                    self.transcript_text.delete("1.0", "end")
                    self.transcript_text.insert(
                        "1.0", transcript_path.read_text(encoding="utf-8").strip()
                    )
            finally:
                self._suppress_auto_transcription = False
            self._set_status(f"Opened {Path(path).name}", "success")
        except (RuntimeError, OSError, ValueError) as exc:
            self._set_status(f"Could not open audio: {exc}", "error")

    def _save_reference(self) -> None:
        if self.audio is None or self.busy_operation is not None:
            return
        path = filedialog.asksaveasfilename(
            title="Save reference audio",
            defaultextension=".wav",
            initialfile="voice-reference.wav",
            filetypes=[("WAV audio", "*.wav")],
        )
        if not path:
            return
        try:
            audio = self._reference_slice()
            sf.write(path, audio, REFERENCE_SAMPLE_RATE, subtype="PCM_16")
            transcript = self.transcript_text.get("1.0", "end").strip()
            if transcript:
                Path(path).with_suffix(".txt").write_text(
                    transcript + os.linesep, encoding="utf-8"
                )
            self._set_status(f"Saved {Path(path).name}", "success")
        except (RuntimeError, OSError, ValueError) as exc:
            self._set_status(f"Could not save reference: {exc}", "error")

    def _write_temp_reference(self) -> str:
        if self._temp_reference:
            try:
                os.unlink(self._temp_reference)
            except FileNotFoundError:
                pass
        with tempfile.NamedTemporaryFile(
            prefix="omnivoice-reference-", suffix=".wav", delete=False
        ) as handle:
            self._temp_reference = handle.name
        sf.write(
            self._temp_reference,
            self._reference_slice(),
            REFERENCE_SAMPLE_RATE,
            subtype="PCM_16",
        )
        return self._temp_reference

    def _generate_clone(self) -> None:
        if self.busy_operation is not None:
            return
        transcript = self.transcript_text.get("1.0", "end").strip()
        target_text = self.clone_text.get("1.0", "end").strip()
        if not transcript:
            self._set_status(
                "Enter the exact reference transcript before generating.",
                "error",
            )
            return
        if not target_text:
            self._set_status("Enter text for the cloned voice to speak.", "error")
            return
        try:
            reference_path = self._write_temp_reference()
            steps = int(self.steps_var.get())
        except (RuntimeError, OSError, ValueError) as exc:
            self._set_status(f"Reference is not ready: {exc}", "error")
            return

        language = self.language_var.get().strip() or None
        self.busy_operation = "generation"
        self._set_status(
            "Loading OmniVoice model..."
            if self.model is None
            else "Generating cloned speech..."
        )
        self.progress.grid()
        self.progress.start(12)
        self._update_controls()

        def worker():
            try:
                import torch

                if self.model is None:
                    from omnivoice import OmniVoice

                    self.model = OmniVoice.from_pretrained(
                        self.model_path,
                        device_map=self.inference_device,
                        dtype=torch.float16,
                    )
                audio = self.model.generate(
                    text=target_text,
                    language=language,
                    ref_audio=reference_path,
                    ref_text=transcript,
                    num_step=steps,
                )[0]
                self.events.put(("generated", np.asarray(audio, dtype=np.float32)))
            except (
                ImportError,
                MemoryError,
                OSError,
                RuntimeError,
                ValueError,
            ) as exc:
                self.events.put(("error", "Generation failed", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _save_generated(self) -> None:
        if self.generated_audio is None or self.busy_operation is not None:
            return
        path = filedialog.asksaveasfilename(
            title="Save cloned speech",
            defaultextension=".wav",
            initialfile="cloned-speech.wav",
            filetypes=[("WAV audio", "*.wav")],
        )
        if not path:
            return
        try:
            sf.write(
                path,
                self.generated_audio,
                REFERENCE_SAMPLE_RATE,
                subtype="PCM_16",
            )
            self._set_status(f"Saved {Path(path).name}", "success")
        except (RuntimeError, OSError, ValueError) as exc:
            self._set_status(f"Could not save result: {exc}", "error")

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "recorded":
                    _, audio, peak, rms = event
                    self.busy_operation = None
                    self.undo_stack.clear()
                    self._set_audio(audio)
                    duration = len(audio) / REFERENCE_SAMPLE_RATE
                    level_note = ""
                    if peak < 0.02 or rms < 0.003:
                        level_note = " Audio level is low; raise output volume."
                    self._set_status(
                        f"Captured {duration:.2f}s.{level_note}",
                        "success" if not level_note else "error",
                    )
                elif kind == "playback_done":
                    self.busy_operation = None
                    self._set_status("Playback finished.", "success")
                elif kind == "generated":
                    _, audio = event
                    self.generated_audio = audio
                    self.busy_operation = None
                    self.progress.stop()
                    self.progress.grid_remove()
                    duration = len(audio) / REFERENCE_SAMPLE_RATE
                    self._set_status(
                        f"Generated {duration:.2f}s of cloned speech.",
                        "success",
                    )
                elif kind == "transcribed":
                    _, transcript, request_id = event
                    self.busy_operation = None
                    self.progress.stop()
                    self.progress.grid_remove()
                    if request_id == self._transcription_request_id:
                        self.transcript_text.delete("1.0", "end")
                        self.transcript_text.insert("1.0", transcript)
                        self._set_status(
                            "Whisper transcript is ready; review it for accuracy.",
                            "success",
                        )
                    elif self.auto_transcribe_var.get():
                        self.root.after(
                            0,
                            lambda: self._start_transcription(
                                self._transcription_request_id
                            ),
                        )
                elif kind == "hotkey":
                    _, action = event
                    if action == "record":
                        if self.busy_operation is None:
                            self._start_recording()
                        elif self.busy_operation != "recording":
                            self._set_status(
                                "Record hotkey ignored while another task is active.",
                                "error",
                            )
                    elif action == "stop":
                        if self.busy_operation == "recording":
                            self._stop_active()
                        else:
                            self._set_status(
                                "Stop hotkey pressed with no active recording."
                            )
                    elif action == "play":
                        self._queue_generated_to_mixer()
                elif kind == "mixer_started":
                    self.mixer_status_var.set(
                        "Live: microphone + queued AI audio -> Voicemeeter Out 6 (B1)."
                    )
                    self._set_status("Game microphone mixer is live.", "success")
                elif kind == "mixer_levels":
                    _, microphone_level, generated_level, output_level = event
                    self.microphone_meter.configure(
                        value=min(100.0, microphone_level * 400)
                    )
                    self.generated_meter.configure(
                        value=min(100.0, generated_level * 400)
                    )
                    self.output_meter.configure(value=min(100.0, output_level * 400))
                elif kind == "mixer_error":
                    _, detail = event
                    self.mixer_status_var.set(f"Mixer error: {detail}")
                    self._set_status(f"Mixer error: {detail}", "error")
                elif kind == "mixer_warning":
                    _, detail = event
                    self.mixer_status_var.set(f"Audio warning: {detail}")
                elif kind == "mixer_stopped":
                    _, had_error = event
                    self.microphone_meter.configure(value=0)
                    self.generated_meter.configure(value=0)
                    self.output_meter.configure(value=0)
                    if not had_error:
                        self.mixer_status_var.set(
                            "Mixer stopped. Set CS2 input to "
                            "Voicemeeter Out 6 (B1) when active."
                        )
                        self._set_status("Game microphone mixer stopped.")
                elif kind == "live_voice_ready":
                    _, detail = event
                    delay = detail.partition("delay_ms=")[2]
                    self.live_voice_status_var.set(
                        f"Live clone active ({self.live_voice_active_preset}) "
                        f"-> Voicemeeter Out 6 (delay {delay} ms)."
                    )
                    self._set_status("Live AI voice conversion is active.", "success")
                elif kind == "live_voice_error":
                    _, detail = event
                    self.live_voice_status_var.set(f"Live clone error: {detail}")
                    self._set_status(f"Live clone error: {detail}", "error")
                elif kind == "live_voice_perf":
                    _, infer_ms = event
                    try:
                        self.live_voice_infer_ms = float(infer_ms)
                    except ValueError:
                        self.live_voice_infer_ms = 0.0
                elif kind == "live_voice_overloaded":
                    self.live_voice_status_var.set(
                        f"{self.live_voice_active_preset} is GPU-limited. "
                        "The selected quality remains unchanged; converted "
                        "audio will be buffered and may arrive late."
                    )
                    self._set_status(
                        "Live clone overload detected; quality was not changed.",
                        "error",
                    )
                elif kind == "live_voice_stopped":
                    _, return_code = event
                    self.live_voice_process = None
                    if return_code == 0:
                        self.live_voice_status_var.set(
                            "Live clone stopped. Record/open a reference to restart."
                        )
                        self._set_status("Live AI voice conversion stopped.")
                    else:
                        self.live_voice_status_var.set(
                            f"Live clone exited with code {return_code}. "
                            f"See {SEED_VC_LOG}."
                        )
                        self._set_status(
                            f"Live clone exited with code {return_code}.",
                            "error",
                        )
                elif kind == "error":
                    _, title, detail = event
                    self.busy_operation = None
                    self.progress.stop()
                    self.progress.grid_remove()
                    self._set_status(f"{title}: {detail}", "error")
                self._update_controls()
        except queue.Empty:
            pass
        self.root.after(50, self._poll_events)

    def _set_status(self, text: str, state: str = "normal") -> None:
        self.status_var.set(text)
        style = {
            "error": "Error.Status.TLabel",
            "success": "Success.Status.TLabel",
        }.get(state, "Status.TLabel")
        self.status_label.configure(style=style)

    def _update_controls(self) -> None:
        has_audio = self.audio is not None and bool(len(self.audio))
        has_selection = self.waveform.selection is not None
        idle = self.busy_operation is None
        recording = self.busy_operation == "recording"
        playback = self.busy_operation == "playback"
        mixer_running = self.mixer.running
        live_voice_running = self._live_voice_running()

        self.record_button.configure(
            state=(
                "normal"
                if idle and self.speakers and not live_voice_running
                else "disabled"
            )
        )
        self.stop_button.configure(
            state="normal" if recording or playback else "disabled"
        )
        self.play_button.configure(state="normal" if idle and has_audio else "disabled")
        self.open_button.configure(
            state="normal" if idle and not live_voice_running else "disabled"
        )
        self.detect_record_hotkey_button.configure(
            state="normal" if idle else "disabled"
        )
        self.detect_stop_hotkey_button.configure(state="normal" if idle else "disabled")
        self.detect_play_hotkey_button.configure(state="normal" if idle else "disabled")
        self.save_button.configure(state="normal" if idle and has_audio else "disabled")
        self.keep_button.configure(
            state="normal" if idle and has_audio and has_selection else "disabled"
        )
        self.delete_button.configure(
            state="normal" if idle and has_audio and has_selection else "disabled"
        )
        self.clear_selection_button.configure(
            state="normal" if idle and has_selection else "disabled"
        )
        self.undo_button.configure(
            state="normal" if idle and self.undo_stack else "disabled"
        )
        self.generate_button.configure(
            state=(
                "normal"
                if idle and has_audio and not live_voice_running
                else "disabled"
            )
        )
        self.transcribe_button.configure(
            state=(
                "normal"
                if idle and has_audio and not live_voice_running
                else "disabled"
            )
        )
        self.auto_transcribe_check.configure(
            state="normal" if idle and not live_voice_running else "disabled"
        )
        self.play_generated_button.configure(
            state=(
                "normal" if idle and self.generated_audio is not None else "disabled"
            )
        )
        self.save_generated_button.configure(
            state=(
                "normal" if idle and self.generated_audio is not None else "disabled"
            )
        )
        self.device_combo.configure(state="readonly" if idle else "disabled")
        hotkey_entry_state = "normal" if idle else "disabled"
        self.record_hotkey_entry.configure(state=hotkey_entry_state)
        self.stop_hotkey_entry.configure(state=hotkey_entry_state)
        self.play_hotkey_entry.configure(state=hotkey_entry_state)
        self.apply_hotkeys_button.configure(state="normal" if idle else "disabled")
        mixer_device_state = "disabled" if mixer_running else "readonly"
        self.mixer_mic_combo.configure(state=mixer_device_state)
        self.virtual_output_combo.configure(state=mixer_device_state)
        self.start_mixer_button.configure(
            state=(
                "normal"
                if not mixer_running
                and self.mixer_microphones
                and self.virtual_outputs
                and not live_voice_running
                else "disabled"
            )
        )
        self.stop_mixer_button.configure(
            state="normal" if mixer_running else "disabled"
        )
        self.send_to_mic_button.configure(
            state=(
                "normal"
                if mixer_running and self.generated_audio is not None
                else "disabled"
            )
        )
        self.start_live_voice_button.configure(
            state=(
                "normal"
                if idle
                and has_audio
                and not live_voice_running
                and self.mixer_microphones
                and self.virtual_outputs
                else "disabled"
            )
        )
        self.stop_live_voice_button.configure(
            state="normal" if live_voice_running else "disabled"
        )
        self.live_voice_quality_combo.configure(
            state="disabled" if live_voice_running else "readonly"
        )

    def _close(self) -> None:
        self.record_stop.set()
        self.play_stop.set()
        self.mixer.stop()
        self._force_stop_live_voice()
        self.hotkey_manager.stop()
        if self._temp_reference:
            try:
                os.unlink(self._temp_reference)
            except FileNotFoundError:
                pass
        if self._transcription_after_id is not None:
            self.root.after_cancel(self._transcription_after_id)
        self.root.destroy()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the native OmniVoice capture and cloning studio."
    )
    parser.add_argument(
        "--model",
        default="local/OmniVoice",
        help="Local checkpoint path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device passed to OmniVoice.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="Whisper model used for automatic reference transcription.",
    )
    parser.add_argument(
        "--asr-device",
        default="cuda:0",
        help="Device used for Whisper transcription.",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    root = tk.Tk()
    try:
        OmniVoiceStudio(
            root,
            model_path=args.model,
            device=args.device,
            asr_model=args.asr_model,
            asr_device=args.asr_device,
        )
    except (RuntimeError, OSError, tk.TclError) as exc:
        root.withdraw()
        messagebox.showerror("OmniVoice Studio", str(exc), parent=root)
        root.destroy()
        raise SystemExit(1) from exc
    root.mainloop()


if __name__ == "__main__":
    main()
