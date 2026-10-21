# OmniVoice 🌍

<p align="center">
  <img width="200" height="200" alt="OmniVoice" src="https://zhu-han.github.io/omnivoice/pics/omnivoice.jpg" />
</p>

<p align="center">
  <a href="https://huggingface.co/k2-fsa/OmniVoice"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model"></a>
  &nbsp;
  <a href="https://huggingface.co/spaces/k2-fsa/OmniVoice"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2604.00688"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a>
  &nbsp;
  <a href="https://zhu-han.github.io/omnivoice"><img src="https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=GitHub&style=flat-square"></a>
  &nbsp;
  <a href="https://colab.research.google.com/github/k2-fsa/OmniVoice/blob/master/docs/OmniVoice.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

OmniVoice is a state-of-the-art massively multilingual zero-shot text-to-speech (TTS) model supporting over 600 languages. Built on a novel diffusion language model-style architecture, it generates high-quality speech with superior inference speed, supporting voice cloning and voice design.

> **OmniVoice Studio fork:** This repository also includes a native Windows
> recording, editing, Whisper transcription, CS2 microphone-mixing, and
> zero-shot live-conversion workflow. See [STUDIO-SETUP.md](STUDIO-SETUP.md),
> or clone the repository and run `setup-and-launch.cmd`.

**Contents**: [Key Features](#key-features) | [Installation](#installation) | [Quick Start](#quick-start) | [Python API](#python-api) | [Command-Line Tools](#command-line-tools) | [Training & Evaluation](#training--evaluation) | [Discussion](#discussion--communication) | [Citation](#citation)

## Key Features

- **600+ Languages Supported**: The broadest language coverage among zero-shot TTS models ([full list](docs/languages.md)).
- **Voice Cloning**: State-of-the-art voice cloning quality.
- **Voice Design**: Control voices via assigned speaker attributes (gender, age, pitch, dialect/accent, whisper, etc.).
- **Fine-grained Control**: Non-verbal symbols (e.g., `[laughter]`) and pronunciation correction via pinyin or phonemes.
- **Fast Inference**: RTF as low as 0.025 (40x faster than real-time).
- **Diffusion Language Model-style Architecture**: A clean, streamlined, and scalable design that delivers both quality and speed.

---

## Installation

Choose **one** of the following methods: **pip** or **uv**.

### pip

> We recommend using a fresh virtual environment (e.g., `conda`, `venv`, etc.) to avoid conflicts.

**Step 1**: Install PyTorch

<details>
<summary>NVIDIA GPU</summary>

```bash
# Install pytorch with your CUDA version, e.g.
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```
> See [PyTorch official site](https://pytorch.org/get-started/locally/) for other versions installation.

</details>

<details>
<summary>Apple Silicon</summary>

```bash
pip install torch==2.8.0 torchaudio==2.8.0
```

</details>

<details>
<summary>Intel Arc GPU (XPU)</summary>

Intel Arc GPUs (Alchemist and Battlemage architectures) are supported via PyTorch's XPU backend.

1. Install the [Intel GPU drivers](https://dgpu-docs.intel.com/driver/installation.html) for your OS.

2. Install PyTorch with XPU support from Intel's wheel index:

```bash
pip install torch torchaudio --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

> See [Intel's PyTorch XPU guide](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/) for version-specific instructions.

3. Verify the backend is working:

```bash
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.device_count())"
```

**Notes**:
- `flash_attn` is not available on XPU; the model automatically falls back to SDPA.
- Training with packed sequences (`flex_attention`) has partial XPU support; single-GPU SDPA training should work.
- Tested on Arc A310 (Alchemist, 4 GB) and Arc Pro B50 (Battlemage, 16 GB).

</details>

**Step 2**: Install OmniVoice (choose one)

```bash
# From PyPI (stable release)
pip install omnivoice

# From the latest source on GitHub (no need to clone)
pip install git+https://github.com/k2-fsa/OmniVoice.git

# For development (clone first, editable install)
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
pip install -e .
```

### uv

Clone the repository and sync dependencies:

```bash
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
uv sync
```

> **Tip**: Can use mirror with `uv sync --default-index "https://mirrors.aliyun.com/pypi/simple"`

---

## Quick Start

Try OmniVoice without coding:

- Launch the local web UI: `omnivoice-demo --ip 0.0.0.0 --port 8001`

- Or try it directly on [HuggingFace Space](https://huggingface.co/spaces/k2-fsa/OmniVoice)

- Or run it in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/k2-fsa/OmniVoice/blob/master/docs/OmniVoice.ipynb)

> If you have trouble connecting to HuggingFace when downloading the pre-trained models, set `export HF_ENDPOINT="https://hf-mirror.com"` before running.

For full usage, see the [Python API](#python-api) and [Command-Line Tools](#command-line-tools) sections below.

---

## Python API

OmniVoice supports three generation modes. All features in this section are also available via [command-line tools](#command-line-tools).

### Voice Cloning

Clone a voice from a short reference audio. Provide `ref_audio` and `ref_text`:

```python
from omnivoice import OmniVoice
import soundfile as sf
import torch

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)
# Apple Silicon users: use device_map="mps" instead
# Intel Arc GPU users: use device_map="xpu" instead

audio = model.generate(
    text="Hello, this is a test of zero-shot voice cloning.",
    ref_audio="ref.wav",
    ref_text="Transcription of the reference audio.",
) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.

# If you don't want to input `ref_text` manually, you can directly omit the `ref_text`.
# The model will use Whisper ASR to auto-transcribe it. To use a local copy (or
# a different Whisper model), pass `asr_model_name="..."` to `from_pretrained`.
# To control which device Whisper is loaded on (e.g. another GPU in multi-GPU
# setups, or the CPU), pass `asr_device="cuda:1"` (or `"cpu"`).

sf.write("out.wav", audio[0], 24000)
```

#### Reusing a cloned voice across sessions

Encode the reference audio once, save the resulting prompt, and skip the
audio loading / auto-transcription steps in later sessions:

```python
prompt = model.create_voice_clone_prompt(
    ref_audio="ref.wav", ref_text="Transcription of the reference audio."
)
prompt.save("my_voice.pt")

# Later, in a new session:
from omnivoice import VoiceClonePrompt

prompt = VoiceClonePrompt.load("my_voice.pt")
audio = model.generate(text="Hello again!", voice_clone_prompt=prompt)
```

> **Tips**
>
> - Use a 3–10 seconds reference audio clip. Longer audio slows down inference and may degrade cloning quality.
> - For standard pronunciation, use a reference audio in the **same language** as the target speech. In cross-lingual voice cloning (i.e., the reference audio and target speech are in different languages), the generated speech will carry an accent from the reference audio's language.
> - For better results with Arabic numerals, normalize them to words first (e.g., "123" → "one hundred twenty-three"). You can pass `normalize_text=True` to `generate()` to do this automatically (opt-in; install the extra with `pip install "omnivoice[tn]"`, which pulls in [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)):
>
>   ```python
>   # "I have 2345 apples." is read correctly instead of digit-by-digit.
>   audio = model.generate(text="I have 2345 apples.", normalize_text=True)
>   ```
>
>   Chinese and English use WeTextProcessing; other languages fall back to `num2words` for integers. Inline control syntax (`[laughter]`, `[B EY1 S]`, pinyin tone markers) is preserved. On macOS (Apple Silicon), `pynini` has no wheel — install it via `conda install -c conda-forge pynini` first.
>
> For more tips, see [docs/tips.md](docs/tips.md).

### Voice Design

Describe the desired voice with speaker attributes — no reference audio needed.
Supported attributes: **gender** (male/female), **age** (child to elderly),
**pitch** (very low to very high), **style** (whisper), **English accent**
(American, British, etc.), and **Chinese dialect** (四川话, 陕西话, etc.).
Attributes are comma-separated and freely combinable across categories.

```python
audio = model.generate(
    text="Hello, this is a test of zero-shot voice design.",
    instruct="female, low pitch, british accent",
)
```

> **Note**: The model is primarily trained on the voice cloning task, so voice cloning is the most stable mode. Voice design is trained on Chinese and English data only. It can generalize to other languages, but may produce unstable results for some low-resource languages or edge cases.

See [docs/voice-design.md](docs/voice-design.md) for the full attribute
reference, Chinese equivalents, and usage tips.

### Auto Voice

Let the model choose a voice automatically:

```python
audio = model.generate(text="This is a sentence without any voice prompt.")
```

### Generation Parameters

All above three modes share the same `model.generate()` API. You can further control the generation behavior via keyword arguments:

```python
audio = model.generate(
    text="...",
    num_step=32,  # diffusion steps (or 16 for faster inference)
    speed=1.0,     # speed factor (>1.0 faster, <1.0 slower)
    duration=10.0, # fixed output duration in seconds (overrides speed)
    # ... more options
)
```
See more detailed control in [docs/generation-parameters.md](docs/generation-parameters.md).

### Non-Verbal & Pronunciation Control

OmniVoice supports inline **non-verbal symbols** and **pronunciation correction** within the input text.

**Non-verbal symbols**: Insert tags like `[laughter]` directly in the text to add expressive non-verbal sounds.

```python
audio = model.generate(text="[laughter] You really got me. I didn't see that coming at all.")
```

Supported tags: `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`, `[dissatisfaction-hnn]`.

**Pronunciation control (Chinese)**: Use pinyin with tone numbers to correct specific character pronunciations.

```python
audio = model.generate(text="这批货物打ZHE2出售后他严重SHE2本了，再也经不起ZHE1腾了。")
```

**Pronunciation control (English)**: Use [CMU pronunciation dictionary](https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict.0.7a)  (uppercase, in brackets) to override default English pronunciations.

```python
audio = model.generate(text="He plays the [B EY1 S] guitar while catching a [B AE1 S] fish.")
```

---

## Command-Line Tools

Three CLI entry points are provided. The CLI tools support all features available in the Python API (voice cloning, voice design, auto voice, generation parameters, etc.) — all controlled via command-line arguments.

| Command | Description | Source |
|---|---|---|
| `omnivoice-demo` | Interactive Gradio web demo | [omnivoice/cli/demo.py](omnivoice/cli/demo.py) |
| `omnivoice-infer` | Single-item inference | [omnivoice/cli/infer.py](omnivoice/cli/infer.py) |
| `omnivoice-infer-batch` | Batch inference across multiple GPUs | [omnivoice/cli/infer_batch.py](omnivoice/cli/infer_batch.py) |
| `omnivoice-capture` | Capture a reference clip from speaker loopback | [omnivoice/cli/capture.py](omnivoice/cli/capture.py) |
| `omnivoice-studio` | Native reference editor and cloning workbench | [omnivoice/cli/capture_ui.py](omnivoice/cli/capture_ui.py) |

### Capture a Voice-Cloning Reference

Install the optional capture dependency:

```bash
pip install "omnivoice[capture]"
```

List speaker devices, then capture 3-10 seconds of clean target speech from
the selected system output:

```bash
omnivoice-capture --list-devices
omnivoice-capture \
    --duration 8 \
    --output reference.wav \
    --transcript "Exact words spoken in the reference clip."
```

The command records speaker loopback, converts it to a 24 kHz mono WAV, and
saves the optional transcript beside it as `reference.txt`. Play only the
voice you want to clone during the recording window. Use the captured clip
with `omnivoice-infer --ref_audio reference.wav --ref_text "Exact words..."`.
The default speaker is used unless `--device` selects a listed index, ID, or
unique device-name substring.

For a native desktop workflow with record, stop, playback, waveform clipping,
automatic Whisper transcription, transcript editing, and clone testing, launch:

```bash
omnivoice-studio --model local/OmniVoice --device cuda:0
```

The studio automatically transcribes new recordings and edited selections with
`openai/whisper-large-v3-turbo`. The model downloads on first use and the
result remains editable before voice cloning. Use `--asr-model` or
`--asr-device` to choose another compatible Whisper checkpoint or device.

The default system-wide hotkeys are `Ctrl+Alt+F9` to start recording and
`Ctrl+Alt+F10` to stop. `Ctrl+Alt+F11` sends the latest generated clip to the
game microphone mixer. They remain active while another application or game
has focus, use the Windows `RegisterHotKey` API without injecting input, and
can be changed directly in the studio. Click **Detect** and press the desired
key combination, or type it manually and click **Apply hotkeys**. Hotkey
choices persist in the user's OmniVoice settings.

### Windows Game Microphone Mixer

OmniVoice Studio can mix a live microphone with generated speech and route the
result into a game through a virtual microphone:

```text
Physical microphone --\
                       +--> VoiceMeeter Input --> VoiceMeeter Out 6 (B1) --> game
Generated OmniVoice --/
```

Install the standard VoiceMeeter virtual audio driver, select the physical
microphone and `Voicemeeter Input` in Studio, then click **Start mixer**. In
CS2, select **Voicemeeter Out 6 (VB-Audio Voicemeeter VAIO)** as the
microphone input. This numbered endpoint is the B1 bus exposed by the current
VoiceMeeter driver. Studio configures the B1 route automatically, disables
hardware monitoring to avoid feedback, and provides separate microphone/AI
gain controls, meters, a limiter, a **Send latest clone** button, and a global
play hotkey.

On Windows, VoiceMeeter can be installed with:

```powershell
winget install --id VB-Audio.Voicemeeter --exact
```

### Zero-Shot Live AI Voice Conversion

OmniVoice Studio can launch an isolated Seed-VC real-time engine so a recorded
or opened reference becomes the voice used while speaking live:

1. Record or open 1-30 seconds of an authorized voice.
2. Trim the waveform to clean speech.
3. Click **Start live clone** in the game microphone mixer.
4. Set CS2 microphone input to **Voicemeeter Out 6** and speak normally.

Studio writes the selected reference to Seed-VC, releases its Whisper/TTS GPU
models, starts the headless converter with the selected physical microphone
and `Voicemeeter Input`, and reports the measured conversion delay. Click
**Stop live clone** before recording another reference or returning to normal
microphone mixing.

Choose a live quality preset before starting:

| Preset | Reference used | Guidance / steps | Tradeoff |
|---|---:|---:|---|
| Fast | Up to 3 s | CFG 0.0 / 4 steps | Lowest latency, weakest similarity |
| Balanced | Up to 6 s | CFG 0.4 / 8 steps | Better identity with moderate delay |
| CS2 stable | Up to 6 s | CFG 0.0 / 8 steps | Default; resists foreground-game GPU contention |
| CS2 survival | Up to 3 s | CFG 0.0 / 2 steps | Maximum continuity under severe load |
| High fidelity | Up to 10 s | CFG 0.7 / 12 steps | Closest zero-shot match, highest delay |

For the closest result, use **High fidelity** with 6-15 seconds of one clean,
dry speaker at a healthy level. Remove music, game audio, reverberation,
overlapping voices, long silences, and clipped peaks. Similar source and target
pitch/register also improves perceived identity.

Studio watches Seed-VC's stream for repeated output underruns. If High fidelity
or Balanced becomes starved when CS2 takes GPU focus, Studio automatically
restarts the same reference with **CS2 stable** instead of allowing the virtual
microphone to fall permanently silent. If CS2 stable also overloads, Studio
restarts with **CS2 survival**. Survival mode keeps a larger converted-audio
buffer and preserves delayed AI blocks instead of discarding them. Under severe
load the converted voice can pause and arrive later, but Studio never exposes
the raw microphone.

The live engine is installed separately at `%USERPROFILE%\SeedVC` with Python
3.10 and CUDA 12.8 so its model dependencies cannot conflict with OmniVoice.
It uses the maintained `seed-vc-realtime` fork and the public
`seed-uvit-tat-xlsr-tiny` zero-shot model. No training is required; use only
voices you own or have permission to convert toward.

### Demo

```bash
omnivoice-demo --ip 0.0.0.0 --port 8001
```

Provides a web UI for voice cloning and voice design. See `omnivoice-demo --help` for all options.

### Single Inference

```bash
# Voice Cloning
# ref_text can be omitted (Whisper will auto-transcribe ref_audio to get it).
omnivoice-infer \
    --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech." \
    --ref_audio ref.wav \
    --ref_text "Transcription of the reference audio." \
    --output hello.wav

# Voice Design
omnivoice-infer --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech." \
    --instruct "male, British accent" \
    --output hello.wav

# Auto Voice
omnivoice-infer \
    --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech."\
    --output hello.wav
```

### Batch Inference

`omnivoice-infer-batch` can distribute batch inference across multiple GPUs, designed for large-scale TTS tasks.

```bash
omnivoice-infer-batch \
    --model k2-fsa/OmniVoice \
    --test_list test.jsonl \
    --res_dir results/
```

The test list is a JSONL file where each line is a JSON object:
```json
{"id": "sample_001", "text": "Hello world", "ref_audio": "/path/to/ref.wav", "ref_text": "Reference transcript", "instruct": "female, british accent", "language_id": "en", "duration": 10.0, "speed": 1.0}
```
Only `id` and `text` are mandatory fields. `ref_audio` and `ref_text` are used in voice cloning mode. `instruct` is used in voice design mode. If no reference audio or instruct are provided, the model will generate text in a random voice.

`language_id`, `duration`, and `speed` are optional. `duration` (in seconds) fixes the output length; `speed` controls the speaking rate. If `duration` and `speed` are both provided, `speed` will be ignored.

### FlashInfer Acceleration

Inference can be accelerated ~2-2.9x losslessly with [FlashInfer](https://github.com/flashinfer-ai/flashinfer) kernels (sequence packing for the CFG cond/uncond pair, fused RMSNorm/RoPE/GEMM kernels, and optional CUDA graphs).

**Installation** (NVIDIA GPUs; pick the index matching your CUDA version, e.g. cu128 for PyTorch built with CUDA 12.8):

```bash
pip install flashinfer-python==0.6.15.post1 "flashinfer-jit-cache==0.6.15.post1+cu128" \
    --extra-index-url https://flashinfer.ai/whl/cu128/
```

**Usage** with the batch inference CLI:

```bash
omnivoice-infer-batch \
    --model k2-fsa/OmniVoice \
    --test_list test.jsonl \
    --res_dir results/ \
    --batch_size 8 \
    --enable_flashinfer true
```

or with the Python API:

```python
from omnivoice.models.omnivoice_flashinfer import apply_flashinfer

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float16)
apply_flashinfer(model)                          # throughput / batched inference
apply_flashinfer(model, enable_cuda_graph=True)  # recommended for batch=1 (low latency)
```

CUDA graphs are recommended for single-stream (batch=1) usage, where kernel-launch overhead dominates; at batch >= 4 the plain FlashInfer path is already the fastest configuration.

**Benchmark** (seed-tts zh testset, 2020 samples / 3.3h audio, voice cloning, single H100, fp16, `num_step=32`; Average RTF as reported by `omnivoice-infer-batch`, outputs ASR-verified lossless):

| batch size | baseline | FlashInfer | speedup |
|---|---|---|---|
| 1 | 0.0899 | 0.0430 | 2.1x |
| 1 + CUDA graph | — | 0.0367 | 2.4x |
| 2 | 0.0480 | 0.0245 | 2.0x |
| 4 | 0.0331 | 0.0152 | 2.2x |
| 8 | 0.0298 | **0.0115** | **2.6x** |

---

## Training & Evaluation

See [examples/](examples/) for the complete pipeline — from data preparation to training, evaluation, and finetuning.

---

## Discussion & Communication

You can directly discuss on [GitHub Issues](https://github.com/k2-fsa/OmniVoice/issues).

You can also scan the QR code to join our wechat group or follow our wechat official account.

| Wechat Group | Wechat Official Account |
| ------------ | ----------------------- |
|![wechat](https://k2-fsa.org/zh-CN/assets/pic/wechat_group.jpg) |![wechat](https://k2-fsa.org/zh-CN/assets/pic/wechat_account.jpg) |

---

## Community Projects

OmniVoice is supported by a growing ecosystem of community projects.
Explore them in [Community Projects](docs/community-projects.md).

---

## Citation

```bibtex
@article{zhu2026omnivoice,
      title={OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models},
      author={Zhu, Han and Ye, Lingxuan and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Han, Zhifeng and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2604.00688},
      year={2026}
}
```

---

## Disclaimer

Users are strictly prohibited from using this model for unauthorized voice cloning, voice impersonation, fraud, scams, or any other illegal or unethical activities. All users shall ensure full compliance with applicable local laws, regulations, and ethical standards. The developers assume no liability for any misuse of this model and advocate for responsible AI development and use, encouraging the community to uphold safety and ethical principles in AI research and applications.
