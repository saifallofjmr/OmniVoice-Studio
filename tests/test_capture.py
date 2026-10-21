import numpy as np
import pytest

from omnivoice.cli.capture import _prepare_audio, _resolve_speaker, _validate_audio
from omnivoice.cli.capture_ui import (
    LIVE_VOICE_PRESETS,
    delete_selection,
    detected_hotkey,
    is_physical_microphone_name,
    is_virtual_audio_device_name,
    is_virtual_output_name,
    keep_selection,
    mix_audio_block,
    parse_hotkey,
    transcribe_audio,
)


class _Speaker:
    def __init__(self, name, device_id):
        self.name = name
        self.id = device_id


class _SoundCard:
    def __init__(self):
        self.speakers = [
            _Speaker("Monitor Audio", "monitor-id"),
            _Speaker("USB Headphones", "headphones-id"),
        ]

    def all_speakers(self):
        return self.speakers

    def default_speaker(self):
        return self.speakers[1]


def test_resolve_default_index_id_and_name():
    sc = _SoundCard()

    assert _resolve_speaker(sc, None).id == "headphones-id"
    assert _resolve_speaker(sc, "0").id == "monitor-id"
    assert _resolve_speaker(sc, "headphones-id").id == "headphones-id"
    assert _resolve_speaker(sc, "usb").id == "headphones-id"


def test_resolve_speaker_rejects_missing_device():
    with pytest.raises(ValueError, match="No speaker device matches"):
        _resolve_speaker(_SoundCard(), "missing")


def test_prepare_audio_mixes_to_mono_and_resamples():
    stereo = np.column_stack(
        (
            np.ones(4800, dtype=np.float32),
            np.zeros(4800, dtype=np.float32),
        )
    )

    audio = _prepare_audio(stereo, 48000, 24000)

    assert audio.shape == (2400,)
    assert np.mean(audio) == pytest.approx(0.5, abs=1e-3)


def test_validate_audio_rejects_silence():
    with pytest.raises(RuntimeError, match="recording is silent"):
        _validate_audio(np.zeros(24000, dtype=np.float32))


def test_keep_selection_extracts_requested_range():
    audio = np.arange(100, dtype=np.float32)

    result = keep_selection(audio, 10, (2.0, 5.0))

    np.testing.assert_array_equal(result, audio[20:50])


def test_delete_selection_joins_remaining_audio():
    audio = np.arange(100, dtype=np.float32)

    result = delete_selection(audio, 10, (2.0, 5.0))

    np.testing.assert_array_equal(result, np.concatenate((audio[:20], audio[50:])))


def test_delete_selection_rejects_entire_clip():
    with pytest.raises(ValueError, match="entire clip"):
        delete_selection(np.ones(20, dtype=np.float32), 10, (0.0, 2.0))


def test_transcribe_audio_normalizes_pipeline_result():
    received = {}

    def asr_pipe(audio):
        received.update(audio)
        return {"text": "  automatic transcript  "}

    result = transcribe_audio(asr_pipe, np.ones(20, dtype=np.float32), 24000)

    assert result == "automatic transcript"
    assert received["sampling_rate"] == 24000
    assert received["array"].dtype == np.float32


def test_transcribe_audio_rejects_empty_result():
    with pytest.raises(RuntimeError, match="empty transcript"):
        transcribe_audio(lambda _audio: {"text": " "}, np.ones(20), 24000)


def test_parse_hotkey_normalizes_modifiers_and_function_key():
    hotkey = parse_hotkey("alt + control + f9")

    assert hotkey.text == "Ctrl+Alt+F9"
    assert hotkey.modifiers == 0x0003
    assert hotkey.virtual_key == 0x78


def test_parse_hotkey_supports_single_named_key():
    hotkey = parse_hotkey("F10")

    assert hotkey.text == "F10"
    assert hotkey.modifiers == 0
    assert hotkey.virtual_key == 0x79


def test_detected_hotkey_converts_tk_keysyms():
    assert detected_hotkey({"Alt", "Ctrl"}, "F11").text == "Ctrl+Alt+F11"
    assert detected_hotkey({"Shift"}, "Prior").text == "Shift+PageUp"


def test_detected_hotkey_rejects_unsupported_tk_keysym():
    with pytest.raises(ValueError, match="Unsupported key"):
        detected_hotkey({"Ctrl"}, "KP_Add")


def test_mix_audio_block_applies_gains():
    microphone = np.full(20, 0.1, dtype=np.float32)
    generated = np.full(20, 0.2, dtype=np.float32)

    mixed = mix_audio_block(microphone, generated, 0.5, 1.5)

    np.testing.assert_allclose(mixed, 0.35, atol=1e-6)


def test_mix_audio_block_limits_peak_without_clipping():
    microphone = np.ones(20, dtype=np.float32)
    generated = np.ones(20, dtype=np.float32)

    mixed = mix_audio_block(microphone, generated, 1.0, 1.0)

    assert np.max(np.abs(mixed)) == pytest.approx(0.98)


def test_mix_audio_block_requires_matching_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        mix_audio_block(np.ones(2), np.ones(3), 1.0, 1.0)


def test_virtual_and_physical_device_detection():
    assert is_virtual_output_name("Voicemeeter Input (VB-Audio Voicemeeter VAIO)")
    assert is_virtual_output_name("CABLE Input (VB-Audio Virtual Cable)")
    assert not is_virtual_output_name("Headphones (HyperX Cloud Alpha Wireless)")
    assert is_physical_microphone_name("Microphone (HyperX Cloud Alpha Wireless)")
    assert not is_physical_microphone_name(
        "Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)"
    )
    assert is_virtual_audio_device_name(
        "Voicemeeter VAIO3 Input (VB-Audio Voicemeeter VAIO)"
    )
    assert not is_virtual_audio_device_name("Headphones (HyperX Cloud Alpha Wireless)")


def test_live_voice_presets_trade_latency_for_similarity():
    fast = LIVE_VOICE_PRESETS["Fast"]
    balanced = LIVE_VOICE_PRESETS["Balanced"]
    high = LIVE_VOICE_PRESETS["High fidelity"]

    assert (
        fast["diffusion_steps"] < balanced["diffusion_steps"] < high["diffusion_steps"]
    )
    assert (
        fast["max_prompt_length"]
        < balanced["max_prompt_length"]
        < high["max_prompt_length"]
    )
    assert fast["block_time"] < balanced["block_time"] < high["block_time"]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("Ctrl+Alt", "missing its main key"),
        ("Ctrl+F9+F10", "only one non-modifier"),
        ("Ctrl+F25", "between F1 and F24"),
        ("Ctrl+Mouse4", "Unsupported hotkey"),
    ],
)
def test_parse_hotkey_rejects_invalid_combinations(value, message):
    with pytest.raises(ValueError, match=message):
        parse_hotkey(value)
