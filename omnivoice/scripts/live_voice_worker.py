"""Run Seed-VC real-time conversion as a headless Studio worker."""

import argparse
import ctypes
import os
import sys
import time
from types import SimpleNamespace


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


def main() -> int:
    args = get_parser().parse_args()
    seed_vc_dir = os.path.abspath(args.seed_vc_dir)
    os.chdir(seed_vc_dir)
    sys.path.insert(0, seed_vc_dir)

    try:
        from realtime_vc_engine import RealtimeVCEngine

        if sys.platform == "win32":
            above_normal_priority = 0x00008000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                above_normal_priority,
            )

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
