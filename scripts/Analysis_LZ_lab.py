import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import metrics_analysis as m_a
import seaborn as sns
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import resample


def find_local_minima(signal):
    inverted_signal = -signal  # Invert the signal to use find_peaks
    minima, _ = find_peaks(inverted_signal)
    return minima


def custom_peak_prominence(signal, peaks, local_minima):
    """
    Manually calculate the prominence of peaks based on the nearest valleys (local minima).
    
    Parameters:
    - signal: 1D numpy array of the input signal.
    - peaks: Indices of detected peaks.
    - local_minima: Indices of local minima.
    
    Returns:
    - prominences: Calculated prominences for each peak.
    """
    prominences = []
    
    for peak in peaks:
        # Find the nearest local minima to the left and right of the peak
        pre_minima = local_minima[local_minima < peak]
        post_minima = local_minima[local_minima > peak]
        
        # Get the nearest local minima
        if len(pre_minima) > 0:
            nearest_pre_minima = pre_minima[-1]
        else:
            nearest_pre_minima = 0  # Start of the signal
        
        if len(post_minima) > 0:
            nearest_post_minima = post_minima[0]
        else:
            nearest_post_minima = len(signal) - 1  # End of the signal
        
        # Calculate prominence: peak height minus the higher of the two minima
        min_valley = min(signal[nearest_pre_minima], signal[nearest_post_minima])
        prominence = signal[peak] - min_valley
        prominences.append(prominence)
    
    return np.array(prominences)

def custom_peak_search(signal, min_prominence=0.5, min_distance=100):
    """
    Custom peak search based on prominence and distance between peaks.
    
    Parameters:
    - signal: The input signal (1D array).
    - min_prominence: Minimum prominence for a peak to be considered significant.
    - min_distance: Minimum distance between two peaks to consider them separate.
    
    Returns:
    - final_peaks: The indices of the final detected peaks.
    - final_prominences: The prominences of the final peaks.
    """
    # Step 1: Use find_peaks to detect preliminary peaks
    preliminary_peaks, _ = find_peaks(signal, distance=min_distance)
    local_minima = find_local_minima(signal)

    # Step 2: Calculate prominences of these preliminary peaks
    prominences = custom_peak_prominence(signal, preliminary_peaks,local_minima)
    # Step 4: Filter peaks based on minimum prominence threshold
    final_peaks = preliminary_peaks[prominences >= min_prominence]
    final_prominences = prominences[prominences >= min_prominence]
    
    return np.array(final_peaks), np.array(final_prominences)




def detect_peaks(signal, min_amplitude):
    """
    Detect peaks in the signal above a given minimum amplitude.
    """
    peaks, g = custom_peak_search(signal, min_prominence=min_amplitude, min_distance=100)

    return peaks

def identify_bursts(signal, min_prominence, burst_window=0.1,original_rate=1000):
    """
    Identifies bursts in the signal based on prominence of peaks and burst windows.
    
    Parameters:
    - signal: The input signal.
    - spike_times: The times of detected spikes.
    - min_prominence: Minimum prominence required for a peak to be considered.
    - burst_window: Maximum time (in seconds) between spikes to be considered part of the same burst.
    - original_rate: Sampling rate of the signal.
    
    Returns:
    - bursts: A list of lists, where each sublist contains the indices of spikes in a burst.
    """
    # Convert burst window to sample units
    burst_window_samples = int(burst_window *original_rate)
    
    # Filter spikes based on prominence
    prominent_spikes = detect_peaks(signal, min_prominence)
    prominent_spike_times = prominent_spikes  # Get the actual time indices of the prominent spikes
    #print(peak_prominences(signal, prominent_spike_times)[0])
    bursts = []
    current_burst = []

    for i, spike in enumerate(prominent_spike_times):
        if len(current_burst) == 0:
            current_burst.append(spike)
        else:
            time_diff = spike - current_burst[-1]
            if time_diff <= burst_window_samples:
                current_burst.append(spike)
            else:
                bursts.append(current_burst)
                current_burst = [spike]
    
    # Add the final burst if it exists
    if len(current_burst) > 0:
        bursts.append(current_burst)

    return bursts

def resample_signal(signal, original_fs, target_fs):
    num_samples = int(len(signal) * target_fs / original_fs)
    #if num_samples != target_length:
    signal = resample(signal, num_samples)
    return signal

# Define the path to the folder containing the files
data_folder = "/Users/reva/Documents/Python/SE_DA_FPM/output_csv"
output_dir = os.path.join(data_folder, "output")
os.makedirs(output_dir, exist_ok=True)
sampling_rate =1017
# Keywords for grouping
groups = {
    "Kv4.3 LOF": [],
    "BKCa1.1 LOF": [],
    "Ctrl": []
}

# Assign colors to groups
group_colors = {
    "Kv4.3 LOF": "blue",
    "BKCa1.1 LOF": "green",
    "Ctrl": "red"
}

# Read and group files based on keywords
for filename in os.listdir(data_folder):
    if not filename.endswith(".csv"):
        continue
    
    file_path = os.path.join(data_folder, filename)
    for group in groups:
        if group in filename:
            groups[group].append(file_path)
            break

# Plot relative power densities for each group
plt.figure(figsize=(10, 8))
for group, files in groups.items():
    if not files:
        continue

    all_relative_power = []
    for file_path in files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            if 'df/f' not in df.columns:
                print(f"Column 'df/f' not found in {file_path}. Skipping.")
                continue
            
            # Extract df/f column
            signal = df['df/f'].values
            signal=signal[1000:]
            signal_centered = (signal - np.mean(signal))/np.std(signal)

            # Compute power spectrum
            freqs, power_dB = m_a.compute_power_spectrum_dB(
                signal_centered, sampling_rate, nperseg=4096, max_freq=30
            )
            
            # Compute relative power density
            relative_power = power_dB / np.sum(power_dB)
            all_relative_power.append(power_dB)
            #plt.plot(freqs, relative_power, label=group, color=group_colors[group])

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Calculate average relative power for the group
    if all_relative_power:
        mean_relative_power = np.mean(all_relative_power, axis=0)
        plt.plot(freqs, mean_relative_power, label=group, color=group_colors[group])

# Customize the plot
plt.title("Relative Power Densities by Group")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Relative Power Density")
plt.legend(loc="upper right")
plt.grid(True)

# Save the plot
plot_filename = os.path.join(output_dir, "grouped_relative_power_densities.png")
plt.savefig(plot_filename)
plt.show()
print(f"Plot saved at {plot_filename}")


sampling_rate = 1000  # Hz, adjust if necessary
burst_window_ms = 10  # Window for burst classification, in milliseconds
burst_window = int((burst_window_ms / 1000) * sampling_rate)


ORIGINAL_RATE = 1017.252625
last_minute_samples = sampling_rate * 120  # Samples in the last minute


# Initialize results storage
results = {group: {
    "burst_amplitudes": [], "non_burst_amplitudes": [], "all_amplitudes": [],
    "burst_durations": [], "non_burst_durations": [], "all_durations": [],
    "burst_frequencies": [], "non_burst_frequencies": [], "all_frequencies": []
} for group in groups.keys()}

# Process files group by group
for group, files in groups.items():
    for file_path in files:
        try:
            # Read the CSV
            df = pd.read_csv(file_path)
            if 'df/f' not in df.columns:
                print(f"Column 'df/f' not found in {file_path}. Skipping.")
                continue

            # Get signal and center it
            signal = df['df/f'].values
            signal=resample_signal(signal, ORIGINAL_RATE, 1000)
            signal=signal[1000:]

            signal =  signal+abs(min(signal))
            if len(signal) > last_minute_samples:
                signal = signal[-last_minute_samples:]
            signal = signal - np.mean(signal)
            plt.plot(signal)
            plt.title(f"{file_path}")
            plt.show()
            # Detect peaks
            prominences = 0 # Set your minimum amplitude threshold for peak detection
            peaks = detect_peaks(signal, prominences)
            bursts = []

            # Compute prominences
            prom_values = peak_prominences(signal, peaks)[0]
            bursts = identify_bursts(signal, min_prominence=np.mean(prom_values)+2.*np.std(prom_values), burst_window=burst_window, original_rate=1000)
            print(np.median(prom_values)+np.std(prom_values))
            #print(prominences)
            # Find local minima in the reversed signal
            local_minima = find_local_minima(signal)
            # Classify bursts

            # Characterize events
            for peak in peaks:
                amplitude = signal[peak]
                half_max = amplitude / 2

                # Find half-max points
                left_idx = np.where(signal[:peak] <= half_max)[0]
                left_half_max = left_idx[-1] if left_idx.size > 0 else 0

                right_idx = np.where(signal[peak:] <= half_max)[0]
                right_half_max = peak + right_idx[0] if right_idx.size > 0 else len(signal) - 1

                fwhm = (right_half_max - left_half_max) / sampling_rate  # FWHM in seconds

                if any(peak in burst for burst in bursts):
                    results[group]["burst_amplitudes"].append(amplitude)
                    results[group]["burst_durations"].append(fwhm)
                else:
                    results[group]["non_burst_amplitudes"].append(amplitude)
                    results[group]["non_burst_durations"].append(fwhm)

                results[group]["all_amplitudes"].append(amplitude)
                results[group]["all_durations"].append(fwhm)

            # Compute frequencies
            total_time = len(signal) / sampling_rate  # Total recording time in seconds
            num_bursts = len(bursts)
            num_burst_spikes = sum(len(burst) for burst in bursts)
            num_non_burst_spikes = len(peaks) - num_burst_spikes

            results[group]["burst_frequencies"].append(num_bursts / total_time)
            results[group]["non_burst_frequencies"].append(num_non_burst_spikes / total_time)
            results[group]["all_frequencies"].append(len(peaks) / total_time)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

# Prepare DataFrames for Visualization
data = {
    "Group": [],
    "Event Type": [],
    "Amplitude": [],
    "Duration (s)": [],
    "Frequency (Hz)": []
}

for group, metrics in results.items():
    for event_type, amplitudes in zip(
        ["Burst Amplitude", "Non-Burst Amplitude", "All Amplitude"],
        [metrics["burst_amplitudes"], metrics["non_burst_amplitudes"], metrics["all_amplitudes"]]
    ):
        data["Group"].extend([group] * len(amplitudes))
        data["Event Type"].extend([event_type] * len(amplitudes))
        data["Amplitude"].extend(amplitudes)
        data["Duration (s)"].extend([np.nan] * len(amplitudes))  # Placeholder for durations
        data["Frequency (Hz)"].extend([np.nan] * len(amplitudes))  # Placeholder for frequencies

    for event_type, durations in zip(
        ["Burst Duration", "Non-Burst Duration", "All Duration"],
        [metrics["burst_durations"], metrics["non_burst_durations"], metrics["all_durations"]]
    ):
        data["Group"].extend([group] * len(durations))
        data["Event Type"].extend([event_type] * len(durations))
        data["Amplitude"].extend([np.nan] * len(durations))  # Placeholder for amplitudes
        data["Duration (s)"].extend(durations)
        data["Frequency (Hz)"].extend([np.nan] * len(durations))  # Placeholder for frequencies

    for event_type, frequencies in zip(
        ["Burst Frequency", "Non-Burst Frequency", "All Frequency"],
        [metrics["burst_frequencies"], metrics["non_burst_frequencies"], metrics["all_frequencies"]]
    ):
        data["Group"].extend([group] * len(frequencies))
        data["Event Type"].extend([event_type] * len(frequencies))
        data["Amplitude"].extend([np.nan] * len(frequencies))  # Placeholder for amplitudes
        data["Duration (s)"].extend([np.nan] * len(frequencies))  # Placeholder for durations
        data["Frequency (Hz)"].extend(frequencies)

df = pd.DataFrame(data)

# Plotting
sns.set(style="whitegrid")

# Amplitude Plot
plt.figure(figsize=(12, 6))
sns.boxplot(x="Event Type", y="Amplitude", hue="Group", data=df, palette="Set2")
plt.title("Event Amplitudes Across Groups")
plt.show()

# Duration Plot
plt.figure(figsize=(12, 6))
sns.boxplot(x="Event Type", y="Duration (s)", hue="Group", data=df, palette="Set2")
plt.title("Event Durations Across Groups")
plt.show()

# Frequency Plot
plt.figure(figsize=(12, 6))
sns.boxplot(x="Event Type", y="Frequency (Hz)", hue="Group", data=df, palette="Set2")
plt.title("Event Frequencies Across Groups")
plt.show()