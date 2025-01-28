import os
import sys
import time
import wave
import threading
import struct
import argparse
from typing import Tuple, Optional
import sounddevice as sd
import mido
import soundfile as sf
from pydub import AudioSegment, silence
import numpy as np
import librosa

# Mapping from note names to semitone offsets
NOTE_OFFSETS = {
    'C': 0,
    'C#': 1, 'Db': 1,
    'D': 2,
    'D#': 3, 'Eb': 3,
    'E': 4,
    'F': 5,
    'F#': 6, 'Gb': 6,
    'G': 7,
    'G#': 8, 'Ab': 8,
    'A': 9,
    'A#': 10, 'Bb': 10,
    'B': 11
}

def list_audio_interfaces() -> list:
    print("\nAvailable Audio Interfaces:")
    devices = sd.query_devices()
    input_devices = []
    for idx, device in enumerate(devices):
        if device['max_input_channels'] >= 2:
            input_devices.append((idx, device))
            print(f"{len(input_devices)-1}: {device['name']} - {device['max_input_channels']} input channels")
    return input_devices

def list_midi_inputs() -> list:
    print("\nAvailable MIDI Inputs:")
    midi_inputs = mido.get_input_names()
    for idx, name in enumerate(midi_inputs):
        print(f"{idx}: {name}")
    return midi_inputs

def choose_audio_interface(default: int) -> int:
    audio_devices = list_audio_interfaces()
    if not audio_devices:
        print("No suitable audio input devices found.")
        sys.exit(1)
    if default < len(audio_devices):
        print(f"Default Audio Interface selected: {audio_devices[default][1]['name']}")
        return audio_devices[default][0]
    else:
        print("Default Audio Interface not available.")
    while True:
        try:
            choice = int(input("Choose an available audio interface (enter number): "))
            if 0 <= choice < len(audio_devices):
                return audio_devices[choice][0]
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def choose_stereo_pair(default: str) -> Tuple[int, int]:
    print("\nChoose the stereo pair for recording.")
    print("Enter the left and right channel numbers separated by a hyphen (e.g., 1-2):")
    if default:
        print(f"Default Stereo Pair selected: {default}")
    while True:
        try:
            pair = input(f"Stereo pair [{default}]: ") or default
            left, right = map(int, pair.strip().split('-'))
            if left < 1 or right < 1:
                print("Channel numbers must be positive integers.")
                continue
            return (left - 1, right - 1)  # Zero-based indexing
        except ValueError:
            print("Invalid format. Please enter in the format X-Y (e.g., 1-2).")

def choose_midi_interface(default: int) -> str:
    midi_inputs = list_midi_inputs()
    if not midi_inputs:
        print("No MIDI input devices found.")
        sys.exit(1)
    if default < len(midi_inputs):
        print(f"Default MIDI Interface selected: {midi_inputs[default]}")
        return midi_inputs[default]
    else:
        print("Default MIDI Interface not available.")
    while True:
        try:
            choice = int(input("Choose an available MIDI interface (enter number): "))
            if 0 <= choice < len(midi_inputs):
                return midi_inputs[choice]
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def choose_midi_channel(default: int) -> int:
    print("\nSpecify a MIDI channel (1-16):")
    if default:
        print(f"Default MIDI Channel selected: {default}")
    while True:
        try:
            channel = input(f"MIDI channel [{default}]: ") or str(default)
            channel = int(channel)
            if 1 <= channel <= 16:
                return channel
            else:
                print("Channel must be between 1 and 16.")
        except ValueError:
            print("Please enter a valid number.")

def read_presets(file_path: str) -> list:
    if not os.path.isfile(file_path):
        print(f"Preset file '{file_path}' not found.")
        sys.exit(1)
    with open(file_path, 'r') as f:
        presets = [line.strip() for line in f if line.strip()]
    if not presets:
        print("No presets found in the file.")
        sys.exit(1)
    return presets

def send_midi_program_change(midi_port: str, channel: int, program_number: int):
    try:
        with mido.open_output(midi_port) as outport:
            # MIDI Program Change messages use 0-based numbering
            program_number = program_number % 128  # Ensure within 0-127
            msg = mido.Message('program_change', program=program_number, channel=channel-1)
            outport.send(msg)
            print(f"Sent Program Change to program number {program_number} (Line {program_number +1})")
    except Exception as e:
        print(f"Error sending MIDI Program Change: {e}")

def play_midi_note(midi_port: str, channel: int, note: int, duration: float, pause_before: float = 2):
    try:
        with mido.open_output(midi_port) as outport:
            time.sleep(pause_before)
            velocity = 100
            msg_on = mido.Message('note_on', note=note, velocity=velocity, channel=channel-1)
            outport.send(msg_on)
            print(f"Playing MIDI Note {note} on channel {channel} for {duration} seconds...")
            time.sleep(duration)
            msg_off = mido.Message('note_off', note=note, velocity=0, channel=channel-1)
            outport.send(msg_off)
            print("MIDI Note stopped.")
    except Exception as e:
        print(f"Error playing MIDI note: {e}")

def create_smpl_chunk(loop_start: int, loop_end: int, sample_rate: int) -> bytes:
    # 'smpl' chunk structure
    manufacturer = 0
    product = 0
    sample_period = int(1e9 / sample_rate)
    midi_unity_note = 60
    midi_pitch_fraction = 0
    smpte_format = 0
    smpte_offset = 0
    num_sample_loops = 1
    sampler_data = 0

    # Loop descriptor
    cue_point_id = 0
    loop_type = 0  # Forward loop
    loop_start_sample = loop_start
    loop_end_sample = loop_end
    loop_fraction = 0
    loop_play_count = 0  # Infinite loop

    # Pack the smpl chunk
    smpl_data = struct.pack('<IIIIIIIIII',
                            manufacturer,
                            product,
                            sample_period,
                            midi_unity_note,
                            midi_pitch_fraction,
                            smpte_format,
                            smpte_offset,
                            num_sample_loops,
                            sampler_data,
                            0)  # Reserved

    # Pack the loop descriptor
    loop_data = struct.pack('<IIIIII',
                            cue_point_id,
                            loop_type,
                            loop_start_sample,
                            loop_end_sample,
                            loop_fraction,
                            loop_play_count)

    # Total smpl chunk size
    smpl_chunk_size = 36 + 24  # Header + body + loop

    smpl_chunk = struct.pack('<4sI', b'smpl', smpl_chunk_size) + smpl_data + loop_data
    return smpl_chunk

def insert_smpl_chunk(wav_filepath: str, smpl_chunk: bytes) -> bool:
    try:
        with open(wav_filepath, 'rb') as f:
            data = f.read()

        # Find 'data' chunk
        data_index = data.find(b'data')
        if data_index == -1:
            print("No 'data' chunk found in WAV file.")
            return False

        # Insert 'smpl' chunk before 'data' chunk
        new_data = data[:data_index] + smpl_chunk + data[data_index:]

        # Update RIFF chunk size
        file_size = len(new_data) - 8  # Exclude 'RIFF' and size field
        new_data = new_data[:4] + struct.pack('<I', file_size) + new_data[8:]

        # Write back to file
        with open(wav_filepath, 'wb') as f:
            f.write(new_data)
        return True
    except Exception as e:
        print(f"Error inserting smpl chunk: {e}")
        return False

def strip_silence(audio_segment: AudioSegment, silence_thresh: int = -50, min_silence_len: int = 100) -> AudioSegment:
    """
    Removes silence from the beginning and end of an AudioSegment.

    Parameters:
    - audio_segment: The AudioSegment to process.
    - silence_thresh: The silence threshold in dBFS. Adjust based on noise levels.
    - min_silence_len: The minimum length of silence to detect in milliseconds.

    Returns:
    - trimmed_audio: The AudioSegment with silence stripped from start and end.
    """
    try:
        # Detect non-silent intervals
        nonsilent_ranges = silence.detect_nonsilent(audio_segment, 
                                                   min_silence_len=min_silence_len, 
                                                   silence_thresh=silence_thresh)
        if not nonsilent_ranges:
            # If no nonsilent parts are found, return the original audio
            return audio_segment

        # Extract the first and last nonsilent ranges
        start_trim = nonsilent_ranges[0][0]
        end_trim = nonsilent_ranges[-1][1]

        # Trim the audio
        trimmed_audio = audio_segment[start_trim:end_trim]
        return trimmed_audio
    except Exception as e:
        print(f"Error in strip_silence: {e}")
        return audio_segment  # Return original if error occurs

def process_audio(filepath: str, fs: int, silence_thresh: int = -50, min_silence_len: int = 100):
    print(f"Processing '{os.path.basename(filepath)}'...")
    try:
        # Load audio with pydub
        audio = AudioSegment.from_wav(filepath)

        # Remove silence using the custom strip_silence function
        trimmed_audio = strip_silence(audio, silence_thresh=silence_thresh, min_silence_len=min_silence_len)

        # Export trimmed audio
        trimmed_audio.export(filepath, format="wav")
        print(f"Silence trimmed for '{os.path.basename(filepath)}'.")

        # Load trimmed audio for analysis
        y, sr = librosa.load(filepath, sr=fs, mono=True)

        # Normalize
        y = y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y

        # Autocorrelation for loop detection
        autocorr = librosa.autocorrelate(y)
        lag = librosa.frames_to_time(np.argmax(autocorr), sr=sr)

        if lag >= 0.5:
            loop_end_sample = int(lag * sr)
            smpl_chunk = create_smpl_chunk(0, loop_end_sample, sr)
            success = insert_smpl_chunk(filepath, smpl_chunk)
            if success:
                print(f"Loop points embedded: 0s to {lag:.2f}s")
            else:
                print("Failed to embed loop points.")
        else:
            print("No suitable loop points found.")

    except Exception as e:
        print(f"Error processing '{filepath}': {e}")

def parse_note(note_str: str) -> Optional[int]:
    """
    Converts a musical note (e.g., C1, D#2) to a MIDI note number.

    Parameters:
    - note_str: The note in string format.

    Returns:
    - MIDI note number if valid, else None.
    """
    try:
        # Split the note into note and octave
        if len(note_str) < 2:
            return None
        if note_str[1] in ['#', 'b']:
            note = note_str[:2]
            octave = int(note_str[2:])
        else:
            note = note_str[0]
            octave = int(note_str[1:])

        semitone = NOTE_OFFSETS.get(note.upper())
        if semitone is None:
            return None

        midi_note = (octave + 1) * 12 + semitone
        if 0 <= midi_note <= 127:
            return midi_note
        else:
            return None
    except:
        return None

def main():
    parser = argparse.ArgumentParser(description="Synthesizer Preset Recorder")
    parser.add_argument('-b', '--bank-name', type=str, help='Name of the Bank')
    parser.add_argument('-ai', '--audio-interface', type=int, default=0, help='Audio Interface index (default: 0)')
    parser.add_argument('-ac', '--audio-channel', type=str, default='1-2', help='Audio Channels (e.g., "1-2", default: "1-2")')
    parser.add_argument('-mi', '--midi-interface', type=int, default=0, help='MIDI Interface index (default: 0)')
    parser.add_argument('-mc', '--midi-channel', type=int, default=1, help='MIDI Channel (1-16, default: 1)')
    parser.add_argument('-n', '--num-patches', type=int, default=None, help='Number of patches to record (default: all)')
    parser.add_argument('-N', '--note', type=str, default='C1', help='MIDI Note to play (e.g., "C1", default: "C1")')
    parser.add_argument('-d', '--duration', type=float, default=12.0, help='Duration of MIDI note in seconds (default: 12)')

    args = parser.parse_args()

    # Bank Name
    bank_name = args.bank_name
    if not bank_name:
        bank_name = input("Enter the name of the Bank: ").strip()
    if not bank_name:
        print("Bank name cannot be empty.")
        sys.exit(1)

    # Patch List File
    patch_file = f"{bank_name}.txt"
    if not os.path.isfile(patch_file):
        print(f"Preset file '{patch_file}' not found.")
        sys.exit(1)

    # Create Bank folder
    os.makedirs(bank_name, exist_ok=True)
    print(f"Folder '{bank_name}' created/existed.")

    # Audio Interface
    audio_interface = choose_audio_interface(args.audio_interface)

    # Audio Channel
    audio_channel = args.audio_channel
    if '-' in audio_channel:
        try:
            left, right = map(int, audio_channel.split('-'))
            stereo_pair = (left - 1, right - 1)  # Zero-based indexing
        except ValueError:
            print("Invalid audio channel format. Using default '1-2'.")
            stereo_pair = (0, 1)
    else:
        print("Invalid audio channel format. Using default '1-2'.")
        stereo_pair = (0, 1)

    # MIDI Interface
    midi_interface_index = args.midi_interface
    midi_port = choose_midi_interface(midi_interface_index)

    # MIDI Channel
    midi_channel = args.midi_channel
    if not (1 <= midi_channel <= 16):
        print("Invalid MIDI channel. Using default channel 1.")
        midi_channel = 1

    # Number of Patches
    num_patches = args.num_patches

    # MIDI Note
    midi_note_str = args.note
    midi_note = parse_note(midi_note_str)
    if midi_note is None:
        print(f"Invalid MIDI note '{midi_note_str}'. Using default note C1.")
        midi_note = 36  # C1
    else:
        print(f"MIDI Note '{midi_note_str}' converted to {midi_note}.")

    # Duration
    duration = args.duration
    if duration <= 0:
        print("Invalid duration. Using default duration of 12 seconds.")
        duration = 12.0

    # Read Presets
    presets = read_presets(patch_file)
    total_patches = len(presets)
    if num_patches is None or num_patches > total_patches:
        num_patches = total_patches
    presets = presets[:num_patches]
    print(f"{len(presets)} presets loaded from '{patch_file}'.")

    # Sample Rate
    fs = 44100  # Sample rate

    for idx, preset in enumerate(presets, start=1):
        print(f"\nRecording preset {idx}: {preset}")

        # Send MIDI Program Change before recording
        send_midi_program_change(midi_port, midi_channel, idx - 1)  # 0-based program number

        # Define file naming
        file_number = f"{idx:03}"
        safe_preset = preset.replace("/", "_").replace("\\", "_")
        filename = f"{file_number} {safe_preset}.wav"
        filepath = os.path.join(bank_name, filename)

        # Start recording in a separate thread
        recording_event = threading.Event()

        def record_audio():
            print("Recording started...")
            try:
                with sf.SoundFile(filepath, mode='w', samplerate=fs, channels=2, subtype='PCM_16') as file:
                    with sd.InputStream(samplerate=fs, device=audio_interface, channels=2, 
                                        callback=lambda indata, frames, time_info, status: file.write(indata)):
                        recording_event.wait()  # Wait until recording is stopped
            except Exception as e:
                print(f"Error during recording: {e}")

        record_thread = threading.Thread(target=record_audio)
        record_thread.start()

        # Play MIDI note
        midi_thread = threading.Thread(target=play_midi_note, args=(midi_port, midi_channel, midi_note, duration))
        midi_thread.start()

        # Record for total duration: pause_before + duration + post_pause
        pause_before = 2  # seconds
        post_pause = 2    # seconds
        total_record_time = pause_before + duration + post_pause
        time.sleep(total_record_time)

        # Stop recording
        recording_event.set()
        record_thread.join()
        midi_thread.join()
        print(f"Recording saved as '{filename}'.")

    # Process WAV Files
    print("\nProcessing recorded WAV files...")
    for idx, preset in enumerate(presets, start=1):
        file_number = f"{idx:03}"
        safe_preset = preset.replace("/", "_").replace("\\", "_")
        filename = f"{file_number} {safe_preset}.wav"
        filepath = os.path.join(bank_name, filename)
        process_audio(filepath, fs, silence_thresh=-50, min_silence_len=100)  # Adjust these parameters as needed

    print("\nAll presets have been recorded and processed.")

if __name__ == "__main__":
    main()
