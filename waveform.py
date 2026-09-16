import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# Load recorded audio
audio, sample_rate = sf.read("data/raw/guitar_test.wav")

# Convert stereo to mono if necessary
if audio.ndim > 1:
    audio = audio[:, 0]

# Create time axis
time = np.arange(len(audio)) / sample_rate

# Plot waveform
plt.figure(figsize=(12, 5))

plt.plot(time, audio)

plt.title("Guitar Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid(True)
plt.tight_layout()
plt.show()
