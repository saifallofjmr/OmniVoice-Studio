"""Run Seed-VC real-time conversion as a headless Studio worker."""

import argparse
import ctypes
import math
import os
import sys
import threading
import time
import types
from types import MethodType, SimpleNamespace

import numpy as np

try:
    from . import live_pitch
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import live_pitch

WHISPER_CHECKPOINT = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
WHISPER_CONFIG = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
F0_CHECKPOINT = "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth"
F0_CONFIG = "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
RMVPE_REPOSITORY = "lj1995/VoiceConversionWebUI"
RMVPE_FILENAME = "rmvpe.pt"


def prepare_reference_audio(
    reference: np.ndarray,
    sample_rate: int,
    max_duration: float,
) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    reference = np.nan_to_num(reference, copy=False)
    if reference.size == 0:
        raise ValueError("The live voice reference is empty.")

    reference = reference - np.mean(reference, dtype=np.float64)
    frame_length = max(1, int(round(sample_rate * 0.02)))
    hop_length = max(1, int(round(sample_rate * 0.01)))
    frame_starts = np.arange(
        0,
        max(1, reference.size - frame_length + 1),
        hop_length,
    )
    frame_rms = np.asarray(
        [
            np.sqrt(
                np.mean(
                    np.square(reference[start : start + frame_length]),
                    dtype=np.float64,
                )
            )
            for start in frame_starts
        ],
        dtype=np.float32,
    )
    threshold = max(1e-4, float(frame_rms.max(initial=0.0)) * 0.04)
    active_frames = np.flatnonzero(frame_rms >= threshold)
    if active_frames.size:
        padding = int(round(sample_rate * 0.05))
        start = max(0, int(frame_starts[active_frames[0]]) - padding)
        end = min(
            reference.size,
            int(frame_starts[active_frames[-1]]) + frame_length + padding,
        )
        reference = reference[start:end]

    max_samples = max(1, int(round(sample_rate * max_duration)))
    reference = reference[:max_samples]
    rms = float(np.sqrt(np.mean(np.square(reference), dtype=np.float64)))
    peak = float(np.max(np.abs(reference), initial=0.0))
    if rms < 1e-5 or peak < 1e-4:
        raise ValueError("The live voice reference contains no usable speech.")

    gain = min(4.0, 0.1 / rms, 0.95 / peak)
    reference = reference * gain
    fade_samples = min(reference.size // 2, int(round(sample_rate * 0.005)))
    if fade_samples:
        fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        reference[:fade_samples] *= fade
        reference[-fade_samples:] *= fade[::-1]
    return np.ascontiguousarray(reference, dtype=np.float32)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-vc-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--input-device", required=True)
    parser.add_argument("--output-device", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=6)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.0)
    parser.add_argument("--max-prompt-length", type=float, default=3.0)
    parser.add_argument("--block-time", type=float, default=0.2)
    parser.add_argument("--extra-time-ce", type=float, default=2.5)
    parser.add_argument("--extra-time", type=float, default=0.5)
    parser.add_argument(
        "--reference-conditioning",
        choices=("raw", "cleaned"),
        default="raw",
    )
    parser.add_argument(
        "--compile-estimator",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--model",
        choices=("tiny", "whisper", "f0"),
        default="tiny",
    )
    parser.add_argument(
        "--prompt-pitch-shift",
        type=float,
        default=0.0,
        help="Formant-preserving semitone shift applied to the reference prompt.",
    )
    parser.add_argument(
        "--register-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Correct octave-register failures by nudging the prompt pitch.",
    )
    return parser


def install_overload_recovery(engine) -> None:
    original_prepare_buffers = engine._prepare_buffers
    last_debt_report = 0.0

    def prepare_larger_rings(self):
        original_prepare_buffers()
        ring_capacity = 32 * self.block_frame
        self._in_ring = np.zeros(ring_capacity, dtype=np.float32)
        self._in_r = 0
        self._in_w = 0
        self._in_avail = 0
        self._out_ring = np.zeros(ring_capacity, dtype=np.float32)
        self._out_r = 0
        self._out_w = self._prime_frames
        self._out_avail = self._prime_frames
        self._out_debt = 0

    def push_delayed_output(self, block):
        nonlocal last_debt_report
        with self._ring_lock:
            preserved_debt = self._out_debt
            self._out_debt = 0
            if self._out_avail + len(block) > len(self._out_ring):
                print("RECOVERY delay_buffer_full", flush=True)
                return
            self._ring_write(self._out_ring, self._out_w, block)
            self._out_w = (self._out_w + len(block)) % len(self._out_ring)
            self._out_avail += len(block)
        if preserved_debt:
            now = time.monotonic()
            if now - last_debt_report >= 1.0:
                last_debt_report = now
                print(
                    f"RECOVERY preserved_delayed_frames={preserved_debt}",
                    flush=True,
                )

    engine._prepare_buffers = MethodType(prepare_larger_rings, engine)
    engine._push_output = MethodType(push_delayed_output, engine)


def condition_engine_on_reference(
    engine,
    reference: np.ndarray,
    f0_conditioner=None,
) -> None:
    import torch
    import torchaudio

    model, semantic_fn, _vocoder, campplus_model, to_mel, _mel_args = engine.model_set
    sample_rate = engine.model_set[-1]["sampling_rate"]
    reference_tensor = torch.from_numpy(reference).to(engine.device)
    reference_16k = torchaudio.functional.resample(reference_tensor, sample_rate, 16000)
    semantic = semantic_fn(reference_16k.unsqueeze(0))
    features = torchaudio.compliance.kaldi.fbank(
        reference_16k.unsqueeze(0),
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    features = features - features.mean(dim=0, keepdim=True)
    engine.style2 = campplus_model(features.unsqueeze(0))
    engine.mel2 = to_mel(reference_tensor.unsqueeze(0))
    target_lengths = engine.mel2.size(2)
    reference_f0 = (
        f0_conditioner.reference_f0(reference, sample_rate)
        if f0_conditioner is not None
        else None
    )
    engine.prompt_condition = model.length_regulator(
        semantic,
        ylens=target_lengths,
        n_quantizers=3,
        f0=reference_f0,
    )[0]
    engine.reference_wav = reference
    engine.reference_wav_name = engine.config.reference_audio_path
    engine.prompt_len = engine.config.max_prompt_length


def precompute_reference_condition(
    engine,
    conditioning: str = "raw",
    pitch_shift: float = 0.0,
    f0_conditioner=None,
) -> SimpleNamespace:
    import librosa

    sample_rate = engine.model_set[-1]["sampling_rate"]
    reference, _ = librosa.load(
        engine.config.reference_audio_path,
        sr=sample_rate,
    )
    if conditioning == "cleaned":
        reference = prepare_reference_audio(
            reference,
            sample_rate,
            engine.config.max_prompt_length,
        )
    else:
        reference = reference[: int(sample_rate * engine.config.max_prompt_length)]
    base_reference = reference
    try:
        pitch = live_pitch.estimate_voiced_pitch(base_reference, sample_rate)
    except (ImportError, ValueError):
        pitch = None
    if pitch_shift and f0_conditioner is None:
        reference = live_pitch.shift_pitch_preserving_formants(
            base_reference, sample_rate, pitch_shift
        )
    if f0_conditioner is not None:
        f0_conditioner.manual_shift = pitch_shift
    condition_engine_on_reference(engine, reference, f0_conditioner)
    reference_rms = float(
        np.sqrt(np.mean(np.square(reference), dtype=np.float64))
    )
    f0_text = (
        f"{pitch.median_hz:.1f}" if pitch is not None and pitch.usable else "unreliable"
    )
    print(
        f"REFERENCE mode={conditioning} "
        f"duration_s={len(reference) / sample_rate:.2f} rms={reference_rms:.3f} "
        f"f0_hz={f0_text} pitch_shift={pitch_shift:+.1f}",
        flush=True,
    )
    info = SimpleNamespace(
        base_reference=base_reference,
        sample_rate=sample_rate,
        pitch=pitch,
        conditioning=conditioning,
        manual_shift=pitch_shift,
        f0_conditioner=f0_conditioner,
    )
    engine.reference_info = info
    return info


class F0Conditioner:
    """Extract and map live F0 into the selected target speaker's register."""

    MAX_REGISTER_SHIFT = 12.0

    def __init__(self, device, *, is_half: bool = True):
        import torch
        from huggingface_hub import hf_hub_download
        from audio_modules.rmvpe import RMVPE

        model_path = hf_hub_download(
            repo_id=RMVPE_REPOSITORY,
            filename=RMVPE_FILENAME,
        )
        self.device = torch.device(device)
        self.rmvpe = RMVPE(
            model_path,
            is_half=is_half,
            device=self.device,
        )
        self.target_hz = float("nan")
        self.manual_shift = 0.0
        self.last_source_hz = float("nan")
        self.last_shift = 0.0

    def _extract(self, audio_16k) -> np.ndarray:
        import torch

        if torch.is_tensor(audio_16k):
            audio_16k = audio_16k.detach()
        else:
            audio_16k = np.asarray(audio_16k, dtype=np.float32)
        f0 = np.asarray(
            self.rmvpe.infer_from_audio(audio_16k, thred=0.03),
            dtype=np.float32,
        )
        return np.nan_to_num(f0, copy=False)

    @staticmethod
    def _voiced_median(f0: np.ndarray) -> float:
        voiced = np.asarray(f0, dtype=np.float32)
        voiced = voiced[(voiced >= 55.0) & (voiced <= 600.0)]
        return float(np.median(voiced)) if voiced.size else float("nan")

    def reference_f0(
        self,
        reference: np.ndarray,
        sample_rate: int,
    ):
        import librosa
        import torch

        reference_16k = librosa.resample(
            np.asarray(reference, dtype=np.float32),
            orig_sr=sample_rate,
            target_sr=16000,
        )
        f0 = self._extract(reference_16k)
        median = self._voiced_median(f0)
        if not math.isfinite(median):
            raise ValueError("The target reference has no reliable pitch.")
        shift = 2.0 ** (self.manual_shift / 12.0)
        f0[f0 > 0] *= shift
        self.target_hz = median * shift
        return torch.from_numpy(f0).unsqueeze(0).to(self.device)

    def source_f0(self, audio_16k, crop_seconds: float):
        import torch

        f0 = self._extract(audio_16k)
        crop_frames = max(0, int(round(crop_seconds * 100.0)))
        if crop_frames:
            f0 = f0[crop_frames:]
        f0, source_hz, shift = live_pitch.map_f0_to_target_register(
            f0,
            self.target_hz,
            max_shift_semitones=self.MAX_REGISTER_SHIFT,
        )
        self.last_source_hz = source_hz
        self.last_shift = shift
        return torch.from_numpy(f0).unsqueeze(0).to(self.device)


class RegisterMonitor:
    """Reports source/clone/target register and nudges failing prompts upward."""

    REPORT_INTERVAL_SECONDS = 2.0

    def __init__(self, engine, info, *, guard_enabled: bool = True):
        self.engine = engine
        self.info = info
        self.sample_rate = engine.config.samplerate
        divisor = math.gcd(live_pitch.ANALYSIS_RATE, self.sample_rate)
        self._up = live_pitch.ANALYSIS_RATE // divisor
        self._down = self.sample_rate // divisor
        self.source_tracker = live_pitch.YinTracker(live_pitch.ANALYSIS_RATE)
        self.output_tracker = live_pitch.YinTracker(live_pitch.ANALYSIS_RATE)
        self.source_history: list[float] = []
        self.output_history: list[float] = []
        self.guard_shift = 0.0
        self.prompt_hz = float("nan")
        self.guard = None
        pitch = info.pitch
        if pitch is not None and pitch.usable:
            self.prompt_hz = pitch.median_hz * 2.0 ** (info.manual_shift / 12.0)
            if guard_enabled:
                self.guard = live_pitch.RegisterGuard(prompt_hz=self.prompt_hz)
        self._pending = None
        self._shift_thread = None
        self._failures = 0
        self._last_report = 0.0
        self._new_frames = 0
        if guard_enabled and self.guard is None:
            print("REGISTER guard=unavailable reason=unreliable_prompt_pitch", flush=True)

    def _resample(self, block: np.ndarray) -> np.ndarray:
        from scipy.signal import resample_poly

        return resample_poly(np.asarray(block, dtype=np.float32), self._up, self._down)

    def before_block(self) -> None:
        pending = self._pending
        if pending is None:
            return
        self._pending = None
        shifted, prompt_hz, total_shift = pending
        condition_engine_on_reference(self.engine, shifted)
        self.prompt_hz = prompt_hz
        self.guard.applied(prompt_hz)
        self.output_history.clear()
        print(
            f"REGISTER guard=applied shift={total_shift:+.1f} target={prompt_hz:.1f}",
            flush=True,
        )

    def after_block(self, indata: np.ndarray, converted: np.ndarray) -> None:
        try:
            source_f0 = self.source_tracker.push(self._resample(indata))
            output_f0 = self.output_tracker.push(self._resample(converted))
            self.source_history.extend(source_f0.tolist())
            self.output_history.extend(output_f0.tolist())
            del self.source_history[:-200]
            del self.output_history[:-200]
            self._new_frames += len(output_f0)
            if self.guard is not None and len(output_f0):
                proposal = self.guard.observe(output_f0)
                if proposal is not None:
                    self._start_shift(proposal)
            self._report()
        except Exception as exc:  # monitoring must never interrupt conversion
            self._failures += 1
            if self._failures == 1:
                print(f"REGISTER monitor_error={exc}", flush=True)

    def _start_shift(self, guard_shift: float) -> None:
        if self._shift_thread is not None and self._shift_thread.is_alive():
            return
        total_shift = self.info.manual_shift + guard_shift
        print(
            f"REGISTER guard=octave_error shift={total_shift:+.1f}",
            flush=True,
        )

        def work() -> None:
            try:
                shifted = live_pitch.shift_pitch_preserving_formants(
                    self.info.base_reference, self.info.sample_rate, total_shift
                )
                measured = live_pitch.estimate_voiced_pitch(
                    shifted, self.info.sample_rate
                )
                prompt_hz = (
                    measured.median_hz
                    if measured.usable
                    else self.info.pitch.median_hz * 2.0 ** (total_shift / 12.0)
                )
                self.guard_shift = guard_shift
                self._pending = (shifted, prompt_hz, total_shift)
            except (ImportError, ValueError, RuntimeError) as exc:
                self.guard.state = "failed"
                print(f"REGISTER guard=failed reason={exc}", flush=True)

        self._shift_thread = threading.Thread(
            target=work, daemon=True, name="RegisterGuardShift"
        )
        self._shift_thread.start()

    def _report(self) -> None:
        now = time.monotonic()
        if self._new_frames < 25 or now - self._last_report < self.REPORT_INTERVAL_SECONDS:
            return
        self._last_report = now
        self._new_frames = 0
        output_hz = float(np.median(self.output_history[-150:]))
        source_hz = (
            float(np.median(self.source_history[-150:]))
            if self.source_history
            else float("nan")
        )
        error = (
            live_pitch.semitones_between(self.prompt_hz, output_hz)
            if math.isfinite(self.prompt_hz)
            else float("nan")
        )
        state = self.guard.state if self.guard is not None else "off"
        print(
            f"REGISTER source={source_hz:.0f} output={output_hz:.0f} "
            f"target={self.prompt_hz:.0f} error={error:+.1f} "
            f"shift={self.info.manual_shift + self.guard_shift:+.1f} guard={state}",
            flush=True,
        )


def install_register_monitor(
    engine,
    info,
    *,
    guard_enabled: bool = True,
) -> RegisterMonitor:
    monitor = RegisterMonitor(engine, info, guard_enabled=guard_enabled)
    original_process_block = engine._process_block

    def process_block(self, indata):
        monitor.before_block()
        converted = original_process_block(indata)
        monitor.after_block(indata, converted)
        return converted

    engine._process_block = MethodType(process_block, engine)
    engine.register_monitor = monitor
    return monitor


def ensure_matplotlib_stub() -> None:
    # BigVGAN's utils import matplotlib for plotting helpers that streaming never calls.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        module = types.ModuleType("matplotlib")
        module.use = lambda *_args, **_kwargs: None
        pylab = types.ModuleType("matplotlib.pylab")
        module.pylab = pylab
        sys.modules["matplotlib"] = module
        sys.modules["matplotlib.pylab"] = pylab


def build_engine_namespace(model: str, gpu: int) -> SimpleNamespace:
    if model == "tiny":
        return SimpleNamespace(
            checkpoint_path=None,
            config_path=None,
            gpu=gpu,
            fp16=True,
        )
    ensure_matplotlib_stub()

    def symlink_unavailable(*_args, **_kwargs):
        raise PermissionError("symlinks are not available for this Windows user")

    os.symlink = symlink_unavailable
    from hf_utils import load_custom_model_from_hf

    if model == "f0":
        checkpoint_name = F0_CHECKPOINT
        config_name = F0_CONFIG
    else:
        checkpoint_name = WHISPER_CHECKPOINT
        config_name = WHISPER_CONFIG
    checkpoint, config = load_custom_model_from_hf(
        "Plachta/Seed-VC",
        checkpoint_name,
        config_name,
    )
    return SimpleNamespace(
        checkpoint_path=checkpoint,
        config_path=config,
        gpu=gpu,
        fp16=True,
    )


def select_stream_devices(input_name: str, output_name: str) -> None:
    import sounddevice as sd

    try:
        host_apis = sd.query_hostapis()
        wasapi_index = next(
            index
            for index, host_api in enumerate(host_apis)
            if host_api["name"] == "Windows WASAPI"
        )
        devices = sd.query_devices()
        input_index = next(
            index
            for index, device in enumerate(devices)
            if device["hostapi"] == wasapi_index
            and device["name"] == input_name
            and device["max_input_channels"] > 0
        )
        output_index = next(
            index
            for index, device in enumerate(devices)
            if device["hostapi"] == wasapi_index
            and device["name"] == output_name
            and device["max_output_channels"] > 0
        )
    except (StopIteration, sd.PortAudioError) as exc:
        raise ValueError(f"Could not resolve WASAPI devices: {exc}") from exc
    sd.default.device = (input_index, output_index)


def warmup_engine(
    engine,
    compile_estimator: bool,
    reference_conditioning: str,
    pitch_shift: float = 0.0,
    f0_conditioner=None,
) -> SimpleNamespace:
    import torch

    model = engine.model_set[0]
    original_estimator = model.cfm.estimator
    compiled = False
    if compile_estimator:
        try:
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraphs = False
            model.cfm.estimator = torch.compile(
                original_estimator,
                fullgraph=False,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            compiled = True
        except (ImportError, OSError, RuntimeError) as exc:
            print(f"COMPILE unavailable: {exc}", flush=True)

    engine._prepare_buffers()
    info = precompute_reference_condition(
        engine,
        reference_conditioning,
        pitch_shift,
        f0_conditioner,
    )
    dummy_input = torch.zeros_like(engine.input_wav_res)
    try:
        engine._custom_infer(dummy_input)
        if engine.device.type == "cuda":
            torch.cuda.synchronize()
        print(
            f"WARMUP estimator={'compiled' if compiled else 'eager'}",
            flush=True,
        )
    except (OverflowError, RuntimeError) as exc:
        if not compiled:
            raise
        print(f"COMPILE failed, reverting to eager: {exc}", flush=True)
        model.cfm.estimator = original_estimator
        engine._custom_infer(dummy_input)
        if engine.device.type == "cuda":
            torch.cuda.synchronize()
        print("WARMUP estimator=eager", flush=True)
    return info


def main() -> int:
    args = get_parser().parse_args()
    if abs(args.prompt_pitch_shift) > live_pitch.MAX_MANUAL_PITCH_SEMITONES:
        print(
            "ERROR --prompt-pitch-shift must be within "
            f"+/-{live_pitch.MAX_MANUAL_PITCH_SEMITONES:g} semitones",
            flush=True,
        )
        return 1
    seed_vc_dir = os.path.abspath(args.seed_vc_dir)
    os.chdir(seed_vc_dir)
    sys.path.insert(0, seed_vc_dir)

    try:
        import torch
        from realtime_vc_engine import RealtimeVCEngine

        if sys.platform == "win32":
            above_normal_priority = 0x00008000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                above_normal_priority,
            )

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        engine = RealtimeVCEngine(build_engine_namespace(args.model, args.gpu))
        print(f"MODEL {args.model}", flush=True)
        f0_conditioner = (
            F0Conditioner(engine.device, is_half=engine.fp16)
            if args.model == "f0"
            else None
        )
        if f0_conditioner is not None:
            engine.f0_fn = f0_conditioner.source_f0
        last_perf_report = 0.0

        def report_performance(infer_time, _delay_time, _measured_delay):
            nonlocal last_perf_report
            now = time.monotonic()
            if infer_time > 0 and now - last_perf_report >= 0.5:
                last_perf_report = now
                print(f"PERF infer_ms={infer_time * 1000:.1f}", flush=True)

        engine.on_perf_update = report_performance
        install_overload_recovery(engine)
        engine.set_config(
            sg_hostapi="Windows WASAPI",
            sg_wasapi_exclusive=False,
            sg_input_device=args.input_device,
            sg_output_device=args.output_device,
            reference_audio_path=os.path.abspath(args.reference),
            sr_type="sr_device",
            diffusion_steps=args.diffusion_steps,
            inference_cfg_rate=args.inference_cfg_rate,
            max_prompt_length=args.max_prompt_length,
            block_time=args.block_time,
            crossfade_time=0.04,
            extra_time_ce=args.extra_time_ce,
            extra_time=args.extra_time,
            extra_time_right=0.02,
        )
        select_stream_devices(args.input_device, args.output_device)
        info = warmup_engine(
            engine,
            args.compile_estimator,
            args.reference_conditioning,
            args.prompt_pitch_shift,
            f0_conditioner,
        )
        if f0_conditioner is not None:
            print(
                f"F0 target_hz={f0_conditioner.target_hz:.1f} "
                "mapping=median_register",
                flush=True,
            )
        install_register_monitor(
            engine,
            info,
            guard_enabled=args.register_guard and f0_conditioner is None,
        )
        engine.start_stream()
        print(f"READY delay_ms={engine.delay_time * 1000:.1f}", flush=True)
        sys.stdin.readline()
        engine.stop_stream()
        print("STOPPED", flush=True)
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
