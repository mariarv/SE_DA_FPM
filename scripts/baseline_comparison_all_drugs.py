import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import peak_prominences, resample
import metrics_analysis as m_a
import seaborn as sns

# Constants
ORIGINAL_RATE = 1017.252625
TARGET_RATE = 1000
SEGMENT_DURATION = 3 * 60  # 3 minutes in seconds

# Paths to the pickle files created previously
pickle_file_path_vs = 'df_combined_vs_all_drugs.pkl'

# Function to detect peaks
def detect_peaks(signal, prominence=0):
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(signal, prominence=prominence)
    return peaks

# Function to identify bursts
def identify_bursts(signal, min_prominence, burst_window, fs):
    peaks = detect_peaks(signal, prominence=min_prominence)
    bursts = []
    current_burst = []
    for i in range(len(peaks)):
        if i == 0 or (peaks[i] - peaks[i-1]) <= burst_window:
            current_burst.append(peaks[i])
        else:
            if current_burst:
                bursts.append(current_burst)
            current_burst = [peaks[i]]
    if current_burst:
        bursts.append(current_burst)
    return bursts

# Function to resample signal
def resample_signal(signal, original_rate, target_rate):
    duration = len(signal) / original_rate
    num_samples = int(duration * target_rate)
    resampled_signal = resample(signal, num_samples)
    return resampled_signal

# Function to segment data
def segment_data(data, fs, condition_name):
    if condition_name == 'opto_drug':
        # Segment from 10 seconds to 3 minutes and 10 seconds
        start_idx = int(10 * fs)
        end_idx = int((SEGMENT_DURATION + 10) * fs)
        segments = {
            'first_3min': data[start_idx:end_idx]
        }
    else:
        # Default segmentation for 'base_after' or other conditions
        segments = {
            'first_3min': data[:int(SEGMENT_DURATION * fs)],
            'second_3min': data[int(SEGMENT_DURATION * fs):int(2 * SEGMENT_DURATION * fs)],
            'last_3min': data[int(-SEGMENT_DURATION * fs):]
        }
    return segments

# Function to process trace and extract the second segment
def process_trace(data, fs, trace_type):
    # Resample to target rate
    resampled_data = resample_signal(data, ORIGINAL_RATE, TARGET_RATE)
    
    if trace_type == 'base_after':
        segments = segment_data(resampled_data, fs, trace_type)
        processed_data = segments['second_3min']  # Extract only the second 3-minute segment
    else:
        processed_data = None

    return processed_data

# Function to process a signal and extract event data
def process_signal_for_events(signal, fs):
    # Preprocess signal
    #signal = m_a.robust_zscore(signal)
    signal = signal - np.mean(signal)
    signal = signal + abs(min(signal))
    # Detect peaks
    peaks = detect_peaks(signal)
    prominences = peak_prominences(signal, peaks)[0]
    # Identify bursts
    burst_window = int(0.1 * fs)  # Adjusted burst window (e.g., 0.1 s)
    bursts = identify_bursts(signal, min_prominence=np.mean(prominences) + 2.*np.std(prominences), burst_window=burst_window, fs=fs)

    # Calculate frequencies
    total_time = len(signal) / fs
    num_bursts = len(bursts)
    num_burst_spikes = sum(len(burst) for burst in bursts)
    non_burst_spikes = [peak for peak in peaks if not any(peak in burst for burst in bursts)]
    num_non_burst_spikes = len(non_burst_spikes)
    num_all_spikes = len(peaks)

    burst_frequency = num_bursts / total_time
    non_burst_frequency = num_non_burst_spikes / total_time
    all_spike_frequency = num_all_spikes / total_time

    # Calculate amplitudes
    burst_amplitudes = []
    non_burst_amplitudes = []
    all_amplitudes = []

    # For each peak
    for peak in peaks:
        amplitude = signal[peak]

        if any(peak in burst for burst in bursts):
            burst_amplitudes.append(amplitude)
        else:
            non_burst_amplitudes.append(amplitude)
        all_amplitudes.append(amplitude)

    return {
        'burst_frequency': burst_frequency,
        'non_burst_frequency': non_burst_frequency,
        'all_spike_frequency': all_spike_frequency,
        'burst_amplitudes': burst_amplitudes,
        'non_burst_amplitudes': non_burst_amplitudes,
        'all_amplitudes': all_amplitudes
    }

# Function to process event data for each animal
def process_event_data(df, drug1, condition_name):
    event_data = []

    # Filter DataFrame for the specific drug
    df_drug = df[df['drug'] == drug1]

    for index, row in df_drug.iterrows():
        if len(row[condition_name]) > 0:
            signal = np.array(row[condition_name])
            processed_signal = process_trace(signal, TARGET_RATE, condition_name)
            if processed_signal is not None:
                event_results = process_signal_for_events(processed_signal, TARGET_RATE)
                event_results['animal_id'] = m_a.get_animal_id(row['file'])
                event_results['drug'] = row['drug']
                event_results['condition'] = condition_name
                event_data.append(event_results)

    # Convert event_data to DataFrame
    event_df = pd.DataFrame(event_data)
    return event_df

# Function to prepare amplitude data for plotting
def prepare_amplitude_data(event_df):
    # Create lists to store data
    event_types = []
    amplitudes = []

    for idx, row in event_df.iterrows():
        # For burst amplitudes
        for amp in row['burst_amplitudes']:
            event_types.append('Burst Amplitude')
            amplitudes.append(amp)
        # For non-burst amplitudes
        for amp in row['non_burst_amplitudes']:
            event_types.append('Non-Burst Amplitude')
            amplitudes.append(amp)
        # For all amplitudes
        for amp in row['all_amplitudes']:
            event_types.append('All Amplitude')
            amplitudes.append(amp)
    
    amplitude_data = {
        "Event Type": event_types,
        "Amplitude": amplitudes
    }
    df_amplitude = pd.DataFrame(amplitude_data)
    return df_amplitude

# Function to prepare frequency data for plotting
def prepare_frequency_data(event_df):
    freq_data = {
        "Event Type": ["Burst Frequency"] * len(event_df) + 
                      ["Non-Burst Frequency"] * len(event_df) + 
                      ["All Frequency"] * len(event_df),
        "Frequency (Hz)": list(event_df['burst_frequency']) + list(event_df['non_burst_frequency']) + list(event_df['all_spike_frequency'])
    }
    df_frequency = pd.DataFrame(freq_data)
    return df_frequency

# Function to plot amplitudes and frequencies
def plot_amplitudes_and_frequencies(df_amplitude, df_frequency, drug1):
    # Plotting Amplitudes
    plt.figure(figsize=(12, 6), facecolor='black')
    ax = sns.swarmplot(x="Event Type", y="Amplitude", data=df_amplitude, palette="bright")
    ax.set_title(f"Event Amplitudes for {drug1} in VS (Second Segment)", color='white')
    ax.set_facecolor('black')
    ax.set_xlabel("Event Type", color='white')
    ax.set_ylabel("Amplitude", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.show()

    # Plotting Frequencies
    plt.figure(figsize=(12, 6), facecolor='black')
    ax = sns.swarmplot(x="Event Type", y="Frequency (Hz)", data=df_frequency, palette="bright")
    ax.set_title(f"Event Frequencies for {drug1} in VS (Second Segment)", color='white')
    ax.set_facecolor('black')
    ax.set_xlabel("Event Type", color='white')
    ax.set_ylabel("Frequency (Hz)", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.show()

# Main function to load data, process event data, and plot results
def main(pickle_file_path_vs):
    # Specify the drug of interest
    drug1 = 'Fentanyl'  
    # Load the DataFrame from the pickle file
    print("Loading DataFrame from pickle file...")
    df_vs = m_a.load_dataframe(pickle_file_path_vs)

    # Process event data for 'base_after' condition and second segment
    print(f"Processing event data for VS 'base_after' condition, drug: {drug1}, second segment...")
    event_df_vs = process_event_data(df_vs, drug1, 'base_after')

    # Prepare data for plotting
    df_amplitude = prepare_amplitude_data(event_df_vs)
    df_frequency = prepare_frequency_data(event_df_vs)

    # Plot amplitudes and frequencies
    plot_amplitudes_and_frequencies(df_amplitude, df_frequency, drug1)

# Run the main function
if __name__ == "__main__":
    main(pickle_file_path_vs)
