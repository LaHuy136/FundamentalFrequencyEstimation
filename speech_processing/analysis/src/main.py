import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def compare(file_name: str) -> None:
    # Load the audio signal and its sampling rate
    signal, sr = librosa.load(f'test_signals/{file_name}.wav', sr=None)

    # Load the F0 data from WaveSurfer and Cepstrum method
    f0s_wave_surfer = np.loadtxt(f'speech_processing/analysis/data/wavesurfer/{file_name}.f0', dtype=float)
    f0s_wave_surfer = f0s_wave_surfer[:, 0]
    f0s = np.loadtxt(f'speech_processing/analysis/data/cepstrum_based_pitch_detection/{file_name}.txt', dtype=float)

    # Ensure both F0 arrays have the same length by padding with zeros
    if len(f0s_wave_surfer) < len(f0s):
        f0s_wave_surfer = np.pad(f0s_wave_surfer, (0, len(f0s) - len(f0s_wave_surfer)), 'constant')
    else:
        f0s = np.pad(f0s, (0, len(f0s_wave_surfer) - len(f0s)), 'constant')

    # Create the plot
    plt.figure(figsize=(20, 10))
    plt.suptitle(file_name)

    # First subplot: Plot the signal with x-axis in seconds
    plt.subplot(2, 1, 1)
    plt.title('Signal')
    time_signal = np.arange(len(signal)) / sr  # Convert samples to seconds
    plt.plot(time_signal, signal)
    plt.xlabel('Time (s)')  # x-axis label in seconds
    plt.ylabel('Amplitude')

    # margin between subplots
    plt.subplots_adjust(hspace=0.5)

    # Second subplot: Plot F0 with x-axis in seconds
    plt.subplot(2, 1, 2)
    plt.title('F0s')

    # Convert frame indices to time (seconds)
    frame_step = 0.015  # 15 ms frame step
    x = np.arange(len(f0s_wave_surfer)) * frame_step  # Time in seconds

    plt.scatter(x, f0s_wave_surfer, label='F0s WaveSurfer', s=20, alpha=0.5)
    plt.scatter(x, f0s, label='F0s using cepstral', s=20, alpha=0.5)
    plt.xlabel('Time (s)')  # x-axis label in seconds
    plt.ylabel('F0 (Hz)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    compare('lab_male')