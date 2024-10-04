import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, convolve, peak_prominences
from oasis.functions import deconvolve
import logging
import os
from sklearn.metrics import mean_squared_error
import metrics_analysis as m_a
from scipy.signal import savgol_filter
import seaborn as sns

# Load the dataframe
PICKLE_FILE_PATH_DS = 'data/df_combined_vs.pkl'
TEMPLATE_FILE_PATH = 'data/Templates/After_Cocaine_DS_mean_traces_dff.csv'
ORIGINAL_RATE = 1017.252625

def load_template(template_file_path, dur):
    if not os.path.exists(template_file_path):
        logging.error(f"File not found: {template_file_path}")
        raise FileNotFoundError(f"File not found: {template_file_path}")
    template_ = pd.read_csv(template_file_path)
    template = template_[dur].values
    logging.info(f"Loaded template from {template_file_path}")
    return np.array(template)

# Extract the "after_coc" column
df = pd.read_pickle(PICKLE_FILE_PATH_DS)
signals = df['base_before_coc']

# Load or define the template signal
templates = [load_template(TEMPLATE_FILE_PATH, "25"), load_template(TEMPLATE_FILE_PATH, "1000")]
template_names = ['25 ms stim', '1000 ms stim']

def bayesian_deconvolution(signal):
    # Perform Bayesian deconvolution using OASIS
    deconvolved_signal, _ = deconvolve(signal)
    return signal, deconvolved_signal

def find_event_times(signal, template, threshold_factor=0.8):
    # Perform convolution with reversed template
    deconvolved_signal = convolve(signal, template[::-1], mode='same')
    # Adaptive threshold based on convolution statistics
    threshold = np.max(deconvolved_signal) * threshold_factor
    # Detect peaks where convolution result exceeds the threshold
    event_times = np.where(deconvolved_signal > threshold)[0]
    return event_times, deconvolved_signal

def iterative_event_detection(signal, template, iterations=5, threshold_factor=0.8):
    detected_events = []
    temp_signal = signal.copy()
    
    for i in range(iterations):
        event_times, deconvolved_signal = find_event_times(temp_signal, template, threshold_factor)
        if len(event_times) == 0:
            break
        detected_events.extend(event_times)
        
        # Mask the detected events in the original signal to prevent overlap
        for event_time in event_times:
            start = max(0, event_time - len(template)//2)
            end = min(len(signal), event_time + len(template)//2)
            temp_signal[start:end] = 0
    
    return detected_events

def detect_peaks(signal, min_amplitude):
    """
    Detect peaks in the signal above a given minimum amplitude.
    """
    peaks, g = m_a.custom_peak_search(signal,  min_prominence=min_amplitude, min_distance=100)

    return peaks

def identify_bursts(signal, min_prominence, burst_window=0.1, original_rate=1017):
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
    burst_window_samples = int(burst_window * original_rate)
    
    # Filter spikes based on prominence
    prominent_spikes = detect_peaks(signal, min_prominence)
    prominent_spike_times = prominent_spikes  # Get the actual time indices of the prominent spikes
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


def extract_min_spike_amplitude(signal, peaks, percentile=80):
    """
    Dynamically determine the minimum spike amplitude based on the detected peak amplitudes.
    
    Parameters:
    - signal: The original signal array.
    - peaks: The indices of detected peaks.
    - percentile: The percentile of peak amplitudes to use as the minimum amplitude threshold.
    
    Returns:
    - min_spike_amplitude: The dynamically determined minimum amplitude.
    """
    peak_amplitudes = signal[peaks]
    min_spike_amplitude = np.percentile(peak_amplitudes, percentile)
    return min_spike_amplitude

def scale_signal(signal):
    return signal / np.max(np.abs(signal))

def evaluate_deconvolution(original_signal, deconvolved_signal):
    snr = np.mean(deconvolved_signal) / np.std(deconvolved_signal)
    rmse = np.sqrt(mean_squared_error(original_signal, deconvolved_signal))
    correlation = np.corrcoef(original_signal, deconvolved_signal)[0, 1]
    return snr, rmse, correlation

def recursive_refinement(signal, bursts, spike_times, prominences, prominence_threshold, original_rate):
    """
    Recursively refines a burst by adding spikes that meet the prominence and timing conditions.
    
    Parameters:
    - signal: The input signal.
    - refined_burst: The current burst being refined.
    - spike_times: The indices of detected spikes.
    - prominences: The prominences of each detected spike.
    - prominence_threshold: Spikes with a prominence lower than this will be included in the burst.
    - original_rate: Sampling rate of the signal.
    
    Returns:
    - refined_burst: The recursively refined burst.
    """
    burst_window_ms = 500
    burst_window_samples = int(burst_window_ms / 1000 * original_rate)

    added_spike = False
    refined_burst=bursts
    for burst in refined_burst:
        # Expand the burst window by 500 ms on both sides
        first_spike = burst[0]
        last_spike = burst[-1]

        # Expand the burst window by 500 ms on both sides
        start_idx = max(0, first_spike - burst_window_samples)
        end_idx = min(len(signal), last_spike + 2*burst_window_samples)

        # Try to add more spikes globally from the signal
        for i, spike in enumerate(spike_times):
            if spike not in burst and start_idx <= spike <= end_idx and  abs(signal[burst[-1]]-signal[spike]) < prominence_threshold  :
                burst.append(spike)
                burst = sorted(burst)
                added_spike = True

    # If we added a spike, call this function again recursively for further refinement
    if added_spike:
        refined_burst = recursive_refinement(signal, refined_burst, spike_times, prominences, prominence_threshold, original_rate)
    
    return refined_burst

def refine_and_expand_bursts(signal, bursts, spike_times, prominences, prominence_threshold, original_rate=1017):
    """
    Refines bursts by recursively adding nearby spikes if they meet the prominence condition.
    Expands bursts by adding 500 ms before and after the burst.

    Parameters:
    - signal: The input signal.
    - bursts: The initially detected bursts (each burst is a list of spike indices).
    - spike_times: The indices of detected spikes.
    - prominences: The prominences of each detected spike.
    - prominence_threshold: Spikes with a prominence lower than this will be included in the burst.
    - original_rate: Sampling rate of the signal.
    
    Returns:
    - expanded_bursts: A list of expanded bursts with spikes included based on prominence and extended by 500ms.
    - included_spikes: A list of spike times (indices) that were included in the bursts.
    - excluded_spikes: A list of spike times (indices) that were excluded from the bursts.
    """
    included_spikes = []
    excluded_spikes = []

    # Refine bursts globally
    expanded_bursts = recursive_refinement(signal, bursts, spike_times, prominences, prominence_threshold, original_rate)

    # Track which spikes were included
    for burst in expanded_bursts:
        included_spikes.extend(burst)

    # Exclude spikes that were not added to any burst
    excluded_spikes = [spike for spike in spike_times if spike not in included_spikes]

    return expanded_bursts, included_spikes, excluded_spikes
def plot_trace_with_bursts(signal, bursts, spike_times, excluded_spikes, original_rate, burst_window_ms=500):
    """
    Plot the signal trace and highlight the burst regions and spikes.
    
    Parameters:
    - signal: The input signal array.
    - bursts: A list of bursts (each burst is a list of spike indices).
    - included_spikes: A list of spikes that are included in bursts.
    - excluded_spikes: A list of spikes that are excluded from bursts.
    - original_rate: The sampling rate of the signal.
    - burst_window_ms: Time window around each burst to highlight (-500ms before, +500ms after).
    """
    plt.figure(figsize=(12, 6), facecolor='black')
    time_vector = np.arange(len(signal)) / original_rate

    # Set the background color to black and the trace color to grey
    ax = plt.gca()
    ax.set_facecolor('black')

    # Plot the grey signal trace
    plt.plot(time_vector, signal, color='grey', label='Signal')

    # Highlight bursts as white traces
    for burst in bursts:
        burst = np.sort(burst)
        first_spike = burst[0]
        last_spike = burst[-1]
        
        # Convert burst window to samples
        burst_window_samples = int(burst_window_ms / 1000 * original_rate)
        
        # Calculate the start and end times for the burst highlight
        start_idx = max(0, first_spike - burst_window_samples)
        end_idx = min(len(signal), last_spike + 1 * burst_window_samples)
        
        # Plot the burst part as a white trace on top of the grey trace
        plt.plot(time_vector[start_idx:end_idx], signal[start_idx:end_idx], color='white', lw=2)

    # Plot excluded spikes (in red) for visibility
    plt.plot(time_vector[excluded_spikes], signal[excluded_spikes], "x", color='green', label='Events')

    # Set labels and title
    plt.title('FPM recording before any drug injection', color='white')
    plt.xlabel('Time (s)', color='white')
    plt.ylabel('DF/F', color='white')
    
    # Customize ticks and labels for visibility in the black background
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    # Add a legend
    #plt.legend(facecolor='black', edgecolor='white')
    
    plt.show()

def find_local_minima(signal):
    """
    Find the local minima in the signal.
    
    Parameters:
    - signal: The input signal.
    
    Returns:
    - minima: Indices of local minima in the signal.
    """
    # Use find_peaks with inverted signal to find minima
    minima, _ = find_peaks(-signal)
    return minima

def refine_burst_with_local_minima(signal, bursts, local_minima):
    """
    Refine burst intervals by expanding to the nearest local minima before the first spike and after the last spike.
    
    Parameters:
    - signal: The input signal.
    - bursts: A list of bursts (each burst is a list of spike indices).
    - local_minima: Indices of local minima in the signal.
    
    Returns:
    - refined_bursts: A list of refined bursts with intervals extended to nearest local minima.
    """
    refined_bursts = []
    
    for burst in bursts:
        burst=np.sort(burst)
        first_spike = burst[0]
        last_spike = burst[-1]
        
        # Find the closest local minima before the first spike and after the last spike
        pre_minima = local_minima[local_minima < first_spike]
        post_minima = local_minima[local_minima > last_spike]
        
        if len(pre_minima) > 0:
            refined_start = pre_minima[-1]  # Closest minimum before the first spike
        else:
            refined_start = first_spike  # No minima before, so use the first spike
        
        if len(post_minima) > 0:
            refined_end = post_minima[0]  # Closest minimum after the last spike
        else:
            refined_end = last_spike  # No minima after, so use the last spike
        
        # Append the refined burst interval
        refined_bursts.append((refined_start, refined_end))
    
    return refined_bursts
# Example usage
def merge_nearby_bursts(bursts, min_interburst_time=500, original_rate=1017):
    """
    Merges nearby bursts if the time between consecutive bursts is less than the specified threshold.
    
    Parameters:
    - bursts: A list of bursts (each burst is a list of spike indices).
    - min_interburst_time: Minimum time (in milliseconds) between bursts to be considered separate.
    - original_rate: Sampling rate of the signal.
    
    Returns:
    - merged_bursts: A list of bursts after merging nearby ones.
    """
    merged_bursts = []
    min_interburst_samples = int(min_interburst_time / 1000 * original_rate)
    
    current_burst = bursts[0]  # Start with the first burst

    for i in range(1, len(bursts)):
        previous_last_spike = current_burst[-1]
        next_first_spike = bursts[i][0]
        time_between_bursts = next_first_spike - previous_last_spike

        # Merge bursts if they are closer than the min_interburst_time
        if time_between_bursts <= min_interburst_samples:
            current_burst.extend(bursts[i])
        else:
            merged_bursts.append(current_burst)
            current_burst = bursts[i]

    # Append the final burst
    merged_bursts.append(current_burst)
    
    return merged_bursts


# Initialize lists to store results
evaluation_results = []
all_scaled_signals = []
all_scaled_deconvolved_signals = []
all_detected_event_times = []
burst_firing_rates = []
nonburst_firing_rates = []
busrt_spikes_rates = []
all_firing_rates = []
all_waveforms, all_files=[], []
# Choose which method to use: OASIS, Template Matching, or Spike Simple
use_method = "spike_simple"  # Options: "oasis", "template_matching", "spike_simple"

for signal_idx, signal in enumerate(signals):
    file_name = df["file"].iloc[signal_idx]  # Get the current file name

    if use_method == "oasis":
        # Use OASIS deconvolution
        original_signal, deconvolved_signal = bayesian_deconvolution(signal)
        spike_times = np.where(deconvolved_signal > 0)[0]
    elif use_method == "template_matching":
        # Use iterative event detection with template matching
        template = templates[0]
        detected_event_times = iterative_event_detection(signal, template)
        all_detected_event_times.append(detected_event_times)
        deconvolved_signal = np.zeros_like(signal)  # Placeholder since we don't have a deconvolved signal in this case
        original_signal = signal
        spike_times = detected_event_times
    elif use_method == "spike_simple":
        # Use spike and burst detection
        signal_=signal
        signal=m_a.robust_zscore(signal)
        prominences = 0.1 # Set your minimum amplitude threshold for peak detection
        peaks = detect_peaks(signal, prominences)
        burst_window = int(0.001 * ORIGINAL_RATE)  # Convert time window to indices (e.g., 0.1 s)
        local_minima = find_local_minima(signal)
        prominences = m_a.custom_peak_prominence(signal, peaks,local_minima)
        #print(min_spike_amplitude)
        bursts = identify_bursts(signal, min_prominence=np.median(prominences)+np.std(prominences), burst_window=burst_window, original_rate=ORIGINAL_RATE)
        print(np.median(prominences))
        print(np.mean(prominences))
        print(np.std(prominences))

        refined_bursts, included_spikes, excluded_spikes = refine_and_expand_bursts(signal, bursts, peaks, prominences,np.median(prominences)+np.std(prominences),  original_rate=ORIGINAL_RATE)

        merged_bursts = merge_nearby_bursts(refined_bursts, min_interburst_time=500, original_rate=ORIGINAL_RATE)

    # Step 3: Refine burst intervals based on local minima
        merged_bursts = refine_burst_with_local_minima(signal, merged_bursts, local_minima)
        #plot_trace_with_bursts(signal_, merged_bursts, peaks, excluded_spikes, original_rate=ORIGINAL_RATE, burst_window_ms=1)

        # Flatten bursts into spike times for plotting
        for burst in bursts:
            all_detected_event_times.append(burst)
        
        waveforms = m_a.extract_waveforms(signal_, peaks, ORIGINAL_RATE)
        all_waveforms.append(waveforms)
        all_files.append(file_name)  # Associate each waveform with the current file

        # Plot all the waveforms
        
        # Calculate spike statistics (amplitude, decay rate, width)
        stats = m_a.calculate_spike_statistics_with_next_spike(waveforms, peaks, ORIGINAL_RATE)
        
        # Plot time to next spike vs amplitude
        #m_a.plot_time_to_next_spike_vs_amplitude(stats)
        #m_a.plot_waveforms_with_color(waveforms, stats['width'], ORIGINAL_RATE)

        # Plot the statistics using swarm plots
        #m_a.plot_spike_statistics(stats)
        total_duration=len(signal)/ORIGINAL_RATE
        burst_rate = len(merged_bursts) / total_duration
        nonburst_rate = len(excluded_spikes) / total_duration
        busrt_spikes_rate = len(included_spikes) / total_duration
        all_rate = len(peaks) / total_duration

        burst_firing_rates.append(burst_rate)
        nonburst_firing_rates.append(nonburst_rate)
        busrt_spikes_rates.append(busrt_spikes_rate)
        all_firing_rates.append(all_rate)




    # Scale signals for visualization
    scaled_original_signal = scale_signal(signal)
    #scaled_deconvolved_signal = scale_signal(deconvolved_signal)

    all_scaled_signals.append(scaled_original_signal)
    #all_scaled_deconvolved_signals.append(scaled_deconvolved_signal)
    # Create a time vector for plotting
    time_vector = np.arange(len(signal)) / ORIGINAL_RATE
    """
    # Plot the results for each individual
    plt.figure(figsize=(12, 6))
    plt.plot(time_vector, signal, label='DF/F')
    
    if use_method == "oasis":
        #plt.plot(time_vector, scaled_deconvolved_signal, label='Deconvolved Signal (OASIS)')
        plt.eventplot(time_vector[spike_times], orientation='horizontal', colors='red', lineoffsets=0.5, linelengths=0.2, label='Detected Spikes (OASIS)')
    elif use_method == "template_matching":
        plt.eventplot(time_vector[spike_times], orientation='horizontal', colors='red', lineoffsets=0.5, linelengths=0.2, label='Detected Events (Template Matching)')
    elif use_method == "spike_simple":
        # Plot the detected peaks (spikes)
        plt.plot(time_vector[peaks], signal[peaks], "x")
        # Plot burst markers for "spike_simple"
        for burst in bursts:
            plt.eventplot(time_vector[burst], orientation='horizontal', colors='green', lineoffsets=-0.1, linelengths=0.1)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Scaled Signal')
    plt.title(f'Individual {signal_idx+1} - Method: {use_method}')
    plt.legend()
    plt.show()
    
# Create a dataframe to summarize the evaluation results
evaluation_df = pd.DataFrame({
    'Individual': np.arange(1, len(signals) + 1),
    'SNR': [result[0] for result in evaluation_results],
    'RMSE': [result[1] for result in evaluation_results],
    'Correlation': [result[2] for result in evaluation_results]
})
"""

dat_wave = pd.DataFrame({
    'waveforms': all_waveforms,
    'file': all_files  # Add file column to associate each waveform with its file
})
dat_wave.to_pickle("/Users/reva/Documents/Python/SE_DA_FPM/results/detected_waveforms_before_coc.pkl")

data = {
    #'Burst Firing Rate': burst_firing_rates,
    'Non-Burst Firing Rate': nonburst_firing_rates,
    'Within-Burst Firing Rate': busrt_spikes_rates,
    'All Firing Rate': all_firing_rates
}

# Display the evaluation dataframe
#print(evaluation_df)
df = pd.DataFrame(data)
df_melted = df.melt(var_name='Type', value_name='Firing Rate')

# Plot the swarm plot on a black background
plt.figure(figsize=(10, 6),facecolor="black")
ax = plt.gca()
ax.set_facecolor('black')

sns.swarmplot(x='Type', y='Firing Rate', data=df_melted, palette='viridis')

# Set axis labels and title
plt.xlabel('Event Type', color='white')
plt.ylabel('Firing Rate (Hz)', color='white')
plt.title('Firing Rates Across Recordings', color='white')

# Customize ticks and labels for black background
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Set spines to white
ax.spines['top'].set_color('white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')

plt.savefig("/Users/reva/Documents/Python/SE_DA_FPM/results/all_firing_rates_spikes.pdf")
# Plot evaluation metrics
"""
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(evaluation_df['Individual'], evaluation_df['SNR'])
plt.xlabel('Individual')
plt.ylabel('SNR')
plt.title('Signal-to-Noise Ratio (SNR)')

plt.subplot(1, 3, 2)
plt.plot(evaluation_df['Individual'], evaluation_df['RMSE'])
plt.xlabel('Individual')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error (RMSE)')

plt.subplot(1, 3, 3)
plt.plot(evaluation_df['Individual'], evaluation_df['Correlation'])
plt.xlabel('Individual')
plt.ylabel('Correlation')
plt.title('Correlation Coefficient')

plt.tight_layout()
plt.show()
"""