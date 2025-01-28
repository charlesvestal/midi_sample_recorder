# Synthesizer Preset Recorder

A simple Python tool to automate the recording of synthesizer presets. Easily capture, process, and organize your presets into high-quality WAV files with embedded loop points.

Give it a list of Patch Names in a text file, and it will record each one by one, incrementing an external synth via midi Program Changes. You end up with a set of one-shots for each program on the synth, ready to load into your favorite sampler.

## Features

- **Automated Recording**: Record multiple presets sequentially with minimal setup.
- **MIDI Integration**: Automatically send MIDI Program Change messages to select presets.
- **Customizable Parameters**:
  - Bank name
  - Audio interface and channels
  - MIDI interface and channel
  - Number of patches to record
  - MIDI note selection and duration
- **Audio Processing**:
  - Removes silence from recordings.
  - Detects and embeds loop points for seamless playback.
- **Flexible Usage**: Supports both command-line arguments and interactive prompts.

## Command-Line Arguments

Run the script with the following options to customize its behavior:

| Flag | Description | Default |
|------|-------------|---------|
| `-b`, `--bank-name` | Name of the Bank | Prompted if not specified |
| `-ai`, `--audio-interface` | Audio Interface index | `0` |
| `-ac`, `--audio-channel` | Audio Channels (e.g., "1-2") | `"1-2"` |
| `-mi`, `--midi-interface` | MIDI Interface index | `0` |
| `-mc`, `--midi-channel` | MIDI Channel (1-16) | `1` |
| `-n`, `--num-patches` | Number of patches to record | All listed in the file |
| `-N`, `--note` | MIDI Note to play (e.g., "C1") | `"C1"` |
| `-d`, `--duration` | Duration of MIDI note in seconds | `12.0` |

### Example Usage

```bash
python preset_recorder.py -b "MySynthBank" -ai 1 -ac "3-4" -mi 0 -mc 2 -n 5 -N "D#2" -d 10
