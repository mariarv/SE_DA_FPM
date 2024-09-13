# analysis of the d1/d2 

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
import metrics_analysis as m_a  # Assuming this is a module you have for computing the power spectrum

from scipy.signal import find_peaks

# Provided function to compute spectrum with confidence intervals
def compute_spectrum_with_confidence(segments, fs, nperseg=4096, noverlap=3072, max_freq=30, confidence=0.95):
    # Store all power spectra
    all_psds = []
    
    for segment in segments:
        freqs, power_dB = m_a.compute_power_spectrum_dB(segment, fs, nperseg=nperseg, noverlap=noverlap, max_freq=max_freq)
        all_psds.append(power_dB)
    
    all_psds = np.array(all_psds)
    
    # Calculate mean and standard error
    mean_psd = np.mean(all_psds, axis=0)
    se_psd = sem(all_psds, axis=0)
    
    # Calculate confidence intervals
    h = se_psd * t.ppf((1 + confidence) / 2., len(segments) - 1)
    
    lower_bound = mean_psd - h
    upper_bound = mean_psd + h
    
    return freqs, mean_psd, lower_bound, upper_bound

# Load the pickle files
pkl_file_1 = 'data/combined_continuous_d1.pkl'
pkl_file_2 = 'data/combined_continuous_d2.pkl'

with open(pkl_file_1, 'rb') as f:
    data_group_1 = pickle.load(f)

with open(pkl_file_2, 'rb') as f:
    data_group_2 = pickle.load(f)

#data_group_1 = [array/max(array) for array in data_group_1]
#data_group_2 = [array/max(array) for array in data_group_2]

def create_spike_train(trace, height=None, distance=None, threshold=2):
    mean = np.mean(trace)
    std = np.std(trace)
    threshold = mean + 2 * std  # 2SD criterion
    peaks, _ = find_peaks(trace, height=threshold)
    spike_train = np.zeros_like(trace)
    spike_train[peaks] = 1  # Set the spike points to 1
    return spike_train

# Create spike trains for all traces in Group 1
spike_trains_group_1 = [create_spike_train(trace) for trace in data_group_1]

# Create spike trains for all traces in Group 2
spike_trains_group_2 = [create_spike_train(trace) for trace in data_group_2]

# Sampling frequency, adjust as necessary
fs = 10

# Compute spectra for both groups
freqs_1, mean_psd_1, lower_bound_1, upper_bound_1 = compute_spectrum_with_confidence(spike_trains_group_1, fs)
freqs_2, mean_psd_2, lower_bound_2, upper_bound_2 = compute_spectrum_with_confidence(spike_trains_group_2, fs)

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(14, 15))

# Plot sample traces for Group 1
for trace in spike_trains_group_1[:3]:
    axs[0].plot(trace[:6000])  # Plot the first 1000 points as an example
axs[0].set_title('Sample Traces for Group 1')
axs[0].set_xlabel('Time (samples)')
axs[0].set_ylabel('Amplitude')

# First group spectra
axs[1].plot(freqs_1, mean_psd_1, label='Mean PSD')
axs[1].fill_between(freqs_1, lower_bound_1, upper_bound_1, color='gray', alpha=0.5, label='Confidence Interval')
for psd in spike_trains_group_1:
    freqs, power_dB = m_a.compute_power_spectrum_dB(psd, fs)
    axs[1].plot(freqs, power_dB, alpha=0.3)
axs[1].set_title('Spectra for Group 1')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Power (dB)')
axs[1].legend()

# Plot sample traces for Group 2
for trace in spike_trains_group_2[:3]:
    axs[2].plot(trace[:6000])  # Plot the first 1000 points as an example
axs[2].set_title('Sample Traces for Group 2')
axs[2].set_xlabel('Time (samples)')
axs[2].set_ylabel('Amplitude')

# Second group spectra
axs[3].plot(freqs_2, mean_psd_2, label='Mean PSD')
axs[3].fill_between(freqs_2, lower_bound_2, upper_bound_2, color='gray', alpha=0.5, label='Confidence Interval')
for psd in spike_trains_group_2:
    freqs, power_dB = m_a.compute_power_spectrum_dB(psd, fs)
    axs[3].plot(freqs, power_dB, alpha=0.3)
axs[3].set_title('Spectra for Group 2')
axs[3].set_xlabel('Frequency (Hz)')
axs[3].set_ylabel('Power (dB)')
axs[3].legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(freqs_1,mean_psd_1, label='Mean (Group 1)')
#plt.fill_between(freqs_1, lower_bound_1, upper_bound_1, color='gray', alpha=0.5, label='Confidence Interval')
plt.title('Mean and Confidence Interval for Group 1')


# Plot for Group 2
plt.plot(freqs_2,mean_psd_2, label='Mean (Group 2)')
#plt.fill_between(freqs_2, lower_bound_2, upper_bound_2, color='gray', alpha=0.5, label='Confidence Interval')
plt.title('Mean and Confidence Interval for Group 2')
plt.xlabel('Sample Index')
plt.ylabel('Z-scored Value')
plt.legend()
plt.show()