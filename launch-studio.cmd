@echo off
setlocal
set "STUDIO=%~dp0.venv\Scripts\omnivoice-studio.exe"
if not exist "%STUDIO%" (
  echo OmniVoice Studio is not installed yet.
  echo Run setup-and-launch.cmd first.
  pause
  exit /b 1
)
start "OmniVoice Studio" "%STUDIO%" --model "%~dp0local\OmniVoice" --device cuda:0 --asr-model openai/whisper-large-v3-turbo --asr-device cuda:0

