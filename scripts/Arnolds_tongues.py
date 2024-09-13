import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
#Arnolds tongues 


# Function to compute dominant frequency
def dominant_frequency(time_series, sampling_rate):
    # Compute FFT
    N = len(time_series)
    yf = fft(time_series)
    xf = fftfreq(N, 1 / sampling_rate)
    
    # Only take the positive frequencies
    xf = xf[:N//2]
    yf = 2.0/N * np.abs(yf[:N//2])
    
    # Find the peak in the FFT
    peak_idx = np.argmax(yf)
    return xf[peak_idx]

# Simulated parameters
sampling_rate = 1000  # Hz
time = np.linspace(0, 10, sampling_rate * 10)  # 10 seconds of data

# Two example time series with different driving frequencies
freq1 = 1.0  # Driving frequency for first time series
freq2 = 1.2  # Driving frequency for second time series

time_series1 = np.sin(2 * np.pi * freq1 * time) + 0.5 * np.sin(2 * np.pi * 2 * freq1 * time)
time_series2 = np.sin(2 * np.pi * freq2 * time) + 0.5 * np.sin(2 * np.pi * 2 * freq2 * time)

# Compute the dominant frequencies
dominant_freq1 = dominant_frequency(time_series1, sampling_rate)
dominant_freq2 = dominant_frequency(time_series2, sampling_rate)

# Compute the frequency ratio
frequency_ratio = dominant_freq1 / dominant_freq2

# Simulate varying parameter (e.g., driving frequency)
driving_frequencies = np.linspace(0.5, 2.0, 100)
tongues = []

for driving_freq in driving_frequencies:
    # Generate a time series for each driving frequency
    time_series = np.sin(2 * np.pi * driving_freq * time) + 0.5 * np.sin(2 * np.pi * 2 * driving_freq * time)
    dom_freq = dominant_frequency(time_series, sampling_rate)
    freq_ratio = dom_freq / driving_freq
    tongues.append(freq_ratio)

# Convert tongues data to numpy array for easier plotting
tongues = np.array(tongues)

# Plotting Arnold tongues
plt.figure(figsize=(10, 6))
plt.plot(driving_frequencies, tongues, label='Frequency Ratio')
plt.axhline(y=1, color='r', linestyle='--', label='1:1 Locking')
plt.axhline(y=2, color='g', linestyle='--', label='2:1 Locking')
plt.axhline(y=0.5, color='b', linestyle='--', label='1:2 Locking')
plt.xlabel('Driving Frequency')
plt.ylabel('Frequency Ratio')
plt.title('Arnold Tongues')
plt.legend()
plt.show()
