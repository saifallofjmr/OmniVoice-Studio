"""Run Seed-VC real-time conversion as a headless Studio worker."""

import argparse
import ctypes
import os
import sys
import time
from types import MethodType, SimpleNamespace

import numpy as np


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


def precompute_reference_condition(engine) -> None:
    import librosa
    import torch
    import torchaudio

    model, semantic_fn, _vocoder, campplus_model, to_mel, mel_args = engine.model_set
    sample_rate = mel_args["sampling_rate"]
    reference, _ = librosa.load(
        engine.config.reference_audio_path,
        sr=sample_rate,
    )
    reference = reference[: int(sample_rate * engine.config.max_prompt_length)]
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
    target_lengths = torch.LongTensor([engine.mel2.size(2)]).to(engine.mel2.device)
    engine.prompt_condition = model.length_regulator(
        semantic,
        ylens=target_lengths,
        n_quantizers=3,
        f0=None,
    )[0]
    engine.reference_wav = reference
    engine.reference_wav_name = engine.config.reference_audio_path
    engine.prompt_len = engine.config.max_prompt_length


def main() -> int:
    args = get_parser().parse_args()
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

        engine = RealtimeVCEngine(
            SimpleNamespace(
                checkpoint_path=None,
                config_path=None,
                gpu=args.gpu,
                fp16=True,
            )
        )
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
        precompute_reference_condition(engine)
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
