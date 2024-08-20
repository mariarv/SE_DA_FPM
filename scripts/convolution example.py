import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Generate a time vector
time = np.linspace(0, 10, 1000)

# Generate a signal composed of a single sine wave (1 Hz)
signal = np.sin(2 * np.pi * 1 * time)  # 1 Hz sine wave

# Generate a sine wave template that does not match the signal (5 Hz)
template_length = 200
template_time = np.linspace(0, 2, template_length)
template_signal = np.sin(2 * np.pi * 5 * template_time)  # 5 Hz sine wave

# Perform convolution-based template matching
deconvolved_signal = convolve(signal, template_signal[::-1], mode='same')

# Plot the original signal
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(time, signal, label='Original Signal')
plt.title('Original Signal (1 Hz Sine Wave)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

# Plot the template signal
plt.subplot(3, 1, 2)
plt.plot(template_time, template_signal, label='Template Signal (5 Hz Sine Wave)', color='green')
plt.title('Poorly Matching Template Signal (5 Hz Sine Wave)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

# Plot the convolved signal
plt.subplot(3, 1, 3)
plt.plot(time, deconvolved_signal, label='Deconvolved Signal', color='orange')
plt.title('Convolved Signal with Poor Template (5 Hz Sine Wave)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()
