import numpy as np
from scipy.signal import find_peaks, correlate
from scipy.signal.windows import hann
from scipy.io import wavfile
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def preprocess_signal(signal):
    """
    Normalize the signal to the range [-1, 1].
    """
    if signal.dtype != np.float32 and signal.dtype != np.float64:
        signal = signal.astype(np.float32) / np.max(np.abs(signal))
    return signal

def frame_signal(signal, frame_size, hop_size, window_func=np.hanning):
    """
    Split the signal into overlapping frames.
    """
    window = window_func(frame_size)
    num_frames = 1 + int((len(signal) - frame_size) / hop_size)
    frames = np.stack([signal[i * hop_size : i * hop_size + frame_size] * window
                       for i in range(num_frames)])
    return frames

def compute_autocorrelation(frame):
    """
    Compute the autocorrelation of the frame.
    """
    autocorr = np.correlate(frame, frame, mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # Keep positive lags
    return autocorr

def estimate_pitch_autocorr(autocorr, sample_rate, min_freq=50, max_freq=2000):
    """
    Estimate pitch from autocorrelation by finding the peak within the desired frequency range.
    """
    # Define the range of lags corresponding to the desired frequency range
    min_lag = int(sample_rate / max_freq)
    max_lag = int(sample_rate / min_freq)
    
    # Find peaks in the autocorrelation
    peaks, _ = find_peaks(autocorr[min_lag:max_lag])
    
    if peaks.size == 0:
        return None  # No pitch detected
    
    # Select the peak with the highest autocorrelation value
    peak = peaks[np.argmax(autocorr[min_lag:max_lag][peaks])]
    
    # Convert lag to frequency
    pitch = sample_rate / (peak + min_lag)
    return pitch

def swipe_pitch_estimation(signal, sample_rate, frame_size=1024, hop_size=512):
    """
    Estimate the pitch of the signal using a SWIPE-like algorithm.
    """
    # Preprocess the signal
    signal = preprocess_signal(signal)
    
    # Frame the signal
    frames = frame_signal(signal, frame_size, hop_size, window_func=np.hanning)
    
    # Initialize list to hold pitch estimates
    pitches = []
    
    for frame in frames:
        autocorr = compute_autocorrelation(frame)
        pitch = estimate_pitch_autocorr(autocorr, sample_rate)
        pitches.append(pitch)
    
    # Post-process pitches to smooth and interpolate missing values
    pitches = np.array(pitches)
    
    # Replace None with np.nan for easier processing
    pitches = np.where(pitches == None, np.nan, pitches)
    
    # Simple median filtering to smooth the pitch estimates
    from scipy.signal import medfilt
    pitches = medfilt(pitches, kernel_size=5)
    
    # Interpolate missing values
    nans = np.isnan(pitches)
    if np.any(nans):
        not_nans = ~nans
        indices = np.arange(len(pitches))
        pitches[nans] = np.interp(indices[nans], indices[not_nans], pitches[not_nans])
    
    # Optionally, apply further smoothing or tracking constraints here
    
    return pitches

# Example usage
if __name__ == "__main__":
    # Read the WAV file
    sample_rate, signal = wavfile.read('data/i.wav')
    
    # If stereo, take the first channel
    if signal.ndim > 1:
        signal = signal[:, 0]
    
    # Parameters
    frame_size = 240
    hop_size = 120
    
    # Estimate pitch using SWIPE-like algorithm
    estimated_pitches = swipe_pitch_estimation(signal, sample_rate, frame_size, hop_size)
    
    # Compute the average pitch
    average_pitch = np.mean(estimated_pitches)
    
    print(f"Estimated Average Pitch: {average_pitch:.2f} Hz")
    
    # (Optional) Plot the pitch contour
    import matplotlib.pyplot as plt
    time_axis = np.arange(len(estimated_pitches)) * hop_size / sample_rate
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, estimated_pitches, label='Estimated Pitch')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Pitch Contour')
    plt.legend()
    plt.grid(True)
    plt.show()
