##########################################################################################################
# problem: stadart find_peaks misses peaks whem using prominence
# 26.09 : fixed prominence: standart pyhtong fucntion tkaes max valey, we need the min valey
# 26.09 : add to the functions
#
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
from scipy.signal import resample

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
    print(peak_prominences(signal, prominent_spike_times)[0])
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
    plt.plot(time_vector[excluded_spikes], signal[excluded_spikes], "x", color='red', label='Excluded Spikes')

    # Set labels and title
    plt.title('Signal with Burst Highlighted and Spike Filtering', color='white')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude', color='white')
    
    # Customize ticks and labels for visibility in the black background
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Add a legend
    plt.legend(facecolor='black', edgecolor='white')
    
    plt.show()

def resample_signal(signal, original_fs, target_fs):
    num_samples = int(len(signal) * target_fs / original_fs)
    #if num_samples != target_length:
    signal = resample(signal, num_samples)
    return signal

# Choose which method to use: OASIS, Template Matching, or Spike Simple
use_method = "spike_simple"  # Options: "oasis", "template_matching", "spike_simple"

for signal_idx, signal in enumerate(signals):
    if use_method == "spike_simple":
        # Use spike and burst detection
        signal=resample_signal(signal, ORIGINAL_RATE, 1000)
        signal=signal[132500:140000]
        signal=m_a.robust_zscore(signal)
        signal=signal+abs(min(signal))
        
        prominences = 0 # Set your minimum amplitude threshold for peak detection
        peaks = detect_peaks(signal, prominences)
        burst_window = int(0.001 * 1000)  # Convert time window to indices (e.g., 0.1 s)
        prominences = peak_prominences(signal, peaks)[0]
        #print(min_spike_amplitude)
        bursts = identify_bursts(signal, min_prominence=0.8, burst_window=burst_window, original_rate=1000)
        print(peaks)
        print(prominences)
        # Find local minima in the reversed signal
        local_minima = find_local_minima(signal)

        # For each peak, find the local minima before and after it
        minima_around_peaks = []
        for peak in peaks:
            # Find local minima before and after the peak
            pre_minima = local_minima[local_minima < peak]
            post_minima = local_minima[local_minima > peak]
            
            # If there are minima found, store the closest ones
            if len(pre_minima) > 0:
                nearest_pre_minima = pre_minima[-1]
            else:
                nearest_pre_minima = None
            
            if len(post_minima) > 0:
                nearest_post_minima = post_minima[0]
            else:
                nearest_post_minima = None
            
            minima_around_peaks.append((peak, nearest_pre_minima, nearest_post_minima, signal[peak]))

        # Display the peaks, local minima, and peak amplitudes
        print(minima_around_peaks)

        # Create a time vector for plotting
        time_vector = np.arange(len(signal)) / 1000

        # Plot the results for each individual
        plt.figure(figsize=(12, 6))
        plt.plot(time_vector, signal, label='DF/F')


        # Plot the detected peaks (spikes)
        plt.plot(time_vector[peaks], signal[peaks], "x")
        # Plot burst markers for "spike_simple"
        for burst in bursts:
            plt.eventplot(time_vector[burst], orientation='horizontal', colors='green', lineoffsets=-0.1, linelengths=0.1)

        plt.show()
