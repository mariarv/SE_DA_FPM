import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis
import seaborn as sns 
from scipy.signal import welch, butter, filtfilt, resample
import pickle
from scipy.signal import convolve, find_peaks
from sklearn.metrics import mean_squared_error
import logging
import os 


def plot_raster(spike_trains, dt, max_time_ms=30000):
    """
    spike_trains: A 2D numpy array where each row represents the spike train of a neuron.
                  Each element is binary (1 for spike, 0 for no spike) or a list of spike timestamps.
    dt: Time step used for the simulation in ms or time units.
    max_time_ms: The maximum time to plot (in ms).
    """
    num_neurons = len(spike_trains)
    max_time_idx = int(max_time_ms / dt)  # Convert max_time_ms to index based on dt
    
    # Create a new figure for the raster plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Loop over each neuron and plot its spike times up to the max time
    for neuron_idx, spike_train in enumerate(spike_trains):
        spike_times = np.where(spike_train > 0)[0] * dt  # Extract spike times based on non-zero entries
        spike_times = spike_times[spike_times <= max_time_ms]  # Limit spike times to max_time_ms
        ax.scatter(spike_times, np.ones_like(spike_times) * neuron_idx, s=2, color="black")

    ax.set_title(f'Raster Plot of Neuronal Spiking Activity (First {max_time_ms/1000:.1f} seconds)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.set_ylim(-0.5, num_neurons - 0.5)
    ax.set_xlim(0, max_time_ms)

def plot_calcium_and_spikes(calcium_traces, spike_trains, dt, neuron_idx, start_idx=1000):    
    """
    Plots the calcium trace from the DataFrame and corresponding spike train for a specific neuron.
    
    Parameters:
    ground_truth_df (DataFrame): Contains the calcium traces in the 'Smoothed Bulk ΔF/F' column.
    spike_trains (2D numpy array): Spike trains for all neurons (binary, 1 for spike).
    dt (float): Time step between samples.
    neuron_idx (int): Index of the neuron to plot.
    start_idx (int): Starting index to plot the calcium traces and spike trains (default: 1000).
    """
    # Extract the specific neuron's calcium trace from the DataFrame starting at `start_idx`
    calcium_trace = ground_truth_df['Smoothed Bulk ΔF/F'].values
    
    # Extract the corresponding spike train for this neuron
    spike_train = spike_trains[neuron_idx][start_idx:]
    
    # Create the time axis based on dt and the length of the calcium trace
    time_axis = np.arange(len(calcium_trace)) * dt
    
    # Plot the calcium trace
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_axis, calcium_trace, label=f'Neuron {neuron_idx} Calcium Trace', color='blue')
    
    # Get the spike times (where the spike train is non-zero)
    spike_times = np.where(spike_train > 0)[0] * dt
    
    # Plot the spike train as dots on top of the calcium trace
    ax.scatter(spike_times, calcium_trace[spike_times.astype(int)], color='red', marker='o', label='Spike Events')

    # Customize the plot
    ax.set_title(f'Calcium Trace and Spikes for Neuron {neuron_idx}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Calcium Signal (ΔF/F)')
    ax.legend(loc='upper right')
    plt.show()

def generate_tonic_spike_trains(num_neurons, total_time, dt, tonic_rate, tonic_interval_noise, firing_type, min_isi_time_points):
    """
    Generate spike trains for tonic neurons with different firing rates drawn from a uniform distribution.

    Args:
        num_neurons (int): Number of neurons.
        total_time (float): Total simulation time in seconds.
        dt (float): Time step in seconds.
        tonic_interval_noise (float): Noise as a percentage of the interval.
        firing_type (str): Type of tonic firing ("random", "synchronous", "asynchronous").
        min_isi_time_points (int): Minimum interval between spikes (in time points).

    Returns:
        list: List of spike trains for the tonic neurons.
    """
    time_points = int(total_time / dt)

    # Define variable firing rates from 0.5 Hz to 8 Hz
    firing_rates = np.random.uniform(0.5, 6, num_neurons)

    # Adjust the firing rates to ensure the mean is exactly 4 Hz
    current_mean = np.mean(firing_rates)
    target_mean = 3.0
    firing_rates = firing_rates + (target_mean - current_mean)
    
    spike_trains = []
    jitter=0.05            
    max_firing_rate = 6.0  # Maximum firing rate used for the base train
    max_interval_between_spikes = int(1 / max_firing_rate / dt)
    base_spike_train = np.zeros(time_points)
    current_time = 0

    # Generate the base spike train with the highest firing frequency (8 Hz)
    while current_time < time_points:
        current_time += max_interval_between_spikes
        if current_time < time_points:
            base_spike_train[current_time] = 1

    for i in range(num_neurons):
        # Determine the mean interval between spikes for each neuron
        tonic_rate = firing_rates[i]
        mean_interval_between_spikes = int(1 / tonic_rate / dt)  # Mean time points between spikes
        
        if firing_type == "synchronous":
            # Synchronous firing pattern: Generate base spike train with some noise

            # Create spike trains for each neuron by removing spikes equidistantly based on their firing rate
            max_jitter_points = int(jitter / dt)  # Convert jitter from seconds to time points

            spike_train = np.zeros(time_points)

            # Calculate the interval for keeping spikes equidistantly based on the neuron’s firing rate
            keep_every_nth_spike = int(max_firing_rate / tonic_rate)
            if keep_every_nth_spike ==0 :
                continue
            
            spike_indices = np.where(base_spike_train > 0)[0]

            for idx, spike_time in enumerate(spike_indices):
                # Keep the spike if it's one of the every-nth spikes
                if idx % keep_every_nth_spike == 0:
                    # Introduce consistent jitter
                    jittered_time = spike_time + np.random.randint(-max_jitter_points, max_jitter_points)
                    jittered_time = min(max(jittered_time, 0), time_points - 1)  # Ensure within bounds
                    spike_train[jittered_time] = 1

            spike_trains.append(spike_train)


        elif firing_type == "asynchronous":
            # Asynchronous firing: Each neuron fires independently with variable intervals
            spike_train = np.zeros(time_points)
            current_time = 0
            while current_time < time_points:
                noisy_interval = mean_interval_between_spikes + int(
                    np.random.normal(0, tonic_interval_noise * mean_interval_between_spikes)
                )
                current_time += max(1, noisy_interval)
                if current_time < time_points:
                    spike_train[current_time] = 1
            spike_trains.append(spike_train)

        elif firing_type == "random":
            # Random firing: Each neuron generates spikes with a Poisson process, varying firing rates
            spike_train = np.zeros(time_points)
            spike_prob = tonic_rate * dt  # Probability of a spike per time step
            spikes = np.random.rand(time_points) < spike_prob
            
            # Enforce minimum ISI by post-processing
            spike_times = np.where(spikes)[0]
            valid_spike_times = []
            if len(spike_times) > 0:
                valid_spike_times.append(spike_times[0])
                for spike_time in spike_times[1:]:
                    if spike_time - valid_spike_times[-1] >= min_isi_time_points:
                        valid_spike_times.append(spike_time)
            spike_train[valid_spike_times] = 1
            spike_trains.append(spike_train)

        else:
            raise ValueError("Invalid firing type. Choose 'random', 'synchronous', or 'asynchronous'.")
    
    return spike_trains

def generate_bursting_spike_trains(num_neurons, total_time, dt, burst_rate, inter_spike_interval, firing_type, avg_spikes_per_burst, sd_spikes_per_burst, jitter):
    """
    Generate spike trains for bursting neurons with different firing types.

    Args:
        num_neurons (int): Number of neurons.
        total_time (float): Total simulation time in seconds.
        dt (float): Time step in seconds.
        burst_rate (float): Bursts per second.
        inter_spike_interval (float): Interval between spikes within a burst (in seconds).
        firing_type (str): Type of bursting firing ("random", "synchronous", "asynchronous").
        avg_spikes_per_burst (int): Average number of spikes per burst.
        sd_spikes_per_burst (int): Standard deviation of spikes per burst.
        jitter (float): Maximum jitter (in seconds) to apply for "synchronous" firing.

    Returns:
        list: List of spike trains for the bursting neurons.
    """
    time_points = int(total_time / dt)

    if firing_type == "synchronous":
        # Generate base burst train
        base_spike_train = np.zeros(time_points)
        burst_times = np.random.poisson(burst_rate * dt, time_points)
        burst_indices = np.where(burst_times > 0)[0]
        for burst_start in burst_indices:
            spikes_in_burst = max(1, int(np.random.normal(avg_spikes_per_burst, sd_spikes_per_burst)))
            for j in range(spikes_in_burst):
                spike_time = burst_start + int(j * inter_spike_interval / dt)
                if spike_time < time_points:
                    base_spike_train[spike_time] = 1

        # Create spike trains with slight jitter
        spike_trains = []
        max_jitter_points = int(jitter / dt)  # Convert jitter from seconds to time points
        for i in range(num_neurons):
            spike_train = np.zeros(time_points)
            spike_indices = np.where(base_spike_train > 0)[0]
            for spike_time in spike_indices:
                jittered_time = spike_time + np.random.randint(-max_jitter_points, max_jitter_points)
                jittered_time = min(max(jittered_time, 0), time_points - 1)  # Ensure within bounds
                spike_train[jittered_time] = 1
            spike_trains.append(spike_train)

    elif firing_type == "asynchronous":
        spike_trains = []
        for i in range(num_neurons):
            spike_train = np.zeros(time_points)
            burst_times = np.random.poisson(burst_rate * dt, time_points)
            burst_indices = np.where(burst_times > 0)[0]
            for burst_start in burst_indices:
                spikes_in_burst = max(1, int(np.random.normal(avg_spikes_per_burst, sd_spikes_per_burst)))
                for j in range(spikes_in_burst):
                    spike_time = burst_start + int(j * inter_spike_interval / dt)
                    if spike_time < time_points:
                        spike_train[spike_time] = 1
            spike_trains.append(spike_train)

    elif firing_type == "random":
        spike_trains = []
        for i in range(num_neurons):
            spike_train = np.zeros(time_points)
            for t in range(time_points):
                if np.random.rand() < burst_rate * dt:
                    spikes_in_burst = max(1, int(np.random.normal(avg_spikes_per_burst, sd_spikes_per_burst)))
                    for j in range(spikes_in_burst):
                        spike_time = t + int(j * inter_spike_interval / dt)
                        if spike_time < time_points:
                            spike_train[spike_time] = 1
            spike_trains.append(spike_train)
    
    else:
        raise ValueError("Invalid firing type. Choose 'random', 'synchronous', or 'asynchronous'.")

    return spike_trains

def generate_tonic_to_bursting_spike_trains(num_neurons, total_time, dt, tonic_rate, burst_rate, inter_spike_interval, switch_prob, firing_type, avg_spikes_per_burst, sd_spikes_per_burst, tonic_interval_noise, jitter):
    """
    Generate spike trains for hybrid neurons that normally fire tonically but can switch to bursting.

    Args:
        num_neurons (int): Number of neurons.
        total_time (float): Total simulation time in seconds.
        dt (float): Time step in seconds.
        tonic_rate (float): Firing rate for tonic firing (in Hz).
        burst_rate (float): Bursts per second when bursting.
        inter_spike_interval (float): Interval between spikes within a burst (in seconds).
        switch_prob (float): Probability of switching from tonic to burst mode at each time step.
        firing_type (str): Type of firing ("random", "synchronous", "asynchronous").
        avg_spikes_per_burst (int): Average number of spikes per burst.
        sd_spikes_per_burst (int): Standard deviation of spikes per burst.
        tonic_interval_noise (float): Noise as a percentage of the interval for tonic firing.
        jitter (float): Maximum jitter (in seconds) to apply for "synchronous" firing.

    Returns:
        list: List of spike trains for the hybrid neurons.
    """
    time_points = int(total_time / dt)
    mean_interval_between_spikes = int(1 / tonic_rate / dt)  # Mean time points between spikes

    # Generate tonic firing rates and adjust to target mean

    spike_trains = []
    
    base_spike_train = np.zeros(time_points)
    current_time = 0

    if firing_type == "synchronous":
        # Generate base spike train
        base_burst_train = np.zeros(time_points)
        current_time = 0
        tonic_rate = np.random.uniform(0.5, 3, num_neurons)

        # Adjust the firing rates to ensure the mean is exactly 4 Hz
        current_mean = np.mean(tonic_rate)
        target_mean = 2.0
        tonic_rate = tonic_rate + (target_mean - current_mean)

        while current_time < time_points:
            # Decide whether to burst or tonic fire
            if np.random.rand() < switch_prob:
                # Bursting mode
                burst_times = np.random.poisson(burst_rate * dt, 1)[0]
                spikes_in_burst = max(1, int(np.random.normal(avg_spikes_per_burst, sd_spikes_per_burst)))
                for j in range(spikes_in_burst):
                    spike_time = current_time + int(j * inter_spike_interval / dt)
                    if spike_time < time_points:
                        base_burst_train[spike_time] = 1
                current_time += int(spikes_in_burst * inter_spike_interval / dt)
            else:
                # Tonic firing mode
                noisy_interval = mean_interval_between_spikes + int(
                    np.random.normal(0, tonic_interval_noise * mean_interval_between_spikes)
                )
                current_time += max(1, noisy_interval)  # Ensure at least 1 time step between spikes


        max_jitter_points = int(jitter / dt)  # Convert jitter from seconds to time points

        for i in range(num_neurons):
            neuron_spike_train = np.zeros(time_points)
            current_time = 0

            # Tonic firing based on individual neuron firing rate
            mean_interval_between_spikes = int(1 / tonic_rate[i] / dt)

            while current_time < time_points:
                # Tonic firing mode
                noisy_interval = mean_interval_between_spikes + int(
                    np.random.normal(0, 0.1 * mean_interval_between_spikes)
                )
                current_time += max(1, noisy_interval)
                if current_time < time_points:
                    neuron_spike_train[current_time] = 1

            # Combine individual tonic train with shared burst train
            combined_train = np.logical_or(neuron_spike_train, base_burst_train).astype(int)

            # Apply jitter to the combined train
            spike_indices = np.where(combined_train > 0)[0]
            jittered_spike_train = np.zeros(time_points)

            for spike_time in spike_indices:
                jittered_time = spike_time + np.random.randint(-max_jitter_points, max_jitter_points)
                jittered_time = min(max(jittered_time, 0), time_points - 1)
                jittered_spike_train[jittered_time] = 1

            spike_trains.append(jittered_spike_train)

    elif firing_type == "asynchronous":
        spike_trains = []
        for i in range(num_neurons):
            spike_train = np.zeros(time_points)
            current_time = 0

            while current_time < time_points:
                # Decide whether to burst or tonic fire
                if np.random.rand() < switch_prob:
                    # Bursting mode
                    burst_times = np.random.poisson(burst_rate * dt, 1)[0]
                    spikes_in_burst = max(1, int(np.random.normal(avg_spikes_per_burst, sd_spikes_per_burst)))
                    for j in range(spikes_in_burst):
                        spike_time = current_time + int(j * inter_spike_interval / dt)
                        if spike_time < time_points:
                            spike_train[spike_time] = 1
                    current_time += int(spikes_in_burst * inter_spike_interval / dt)
                else:
                    # Tonic firing mode
                    noisy_interval = mean_interval_between_spikes + int(
                        np.random.normal(0, tonic_interval_noise * mean_interval_between_spikes)
                    )
                    current_time += max(1, noisy_interval)  # Ensure at least 1 time step between spikes
                    if current_time < time_points:
                        spike_train[current_time] = 1

            spike_trains.append(spike_train)

    elif firing_type == "random":
        spike_trains = []
        for i in range(num_neurons):
            spike_train = np.random.poisson(tonic_rate * dt, time_points)
            spike_trains.append(spike_train)
    
    else:
        raise ValueError("Invalid firing type. Choose 'random', 'synchronous', or 'asynchronous'.")

    return spike_trains


# Function to load the DataFrame from a pickle file
def load_dataframe(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        df = pickle.load(f)
    return df

# Function to calculate full width at half maximum (FWHM)
def calculate_fwhm(trace, sampling_rate):
    half_max = np.max(trace) / 2
    indices = np.where(trace >= half_max)[0]
    if len(indices) < 2:
        return np.nan
    fwhm = (indices[-1] - indices[0]) / sampling_rate
    return fwhm

# Function to calculate peak amplitude
def calculate_peak_amplitude(trace):
    return np.max(trace)

# Function to calculate rise rate (from start to peak)
def calculate_rise_rate(trace, sampling_rate):
    peak_idx = np.argmax(trace)
    rise_rate = (trace[peak_idx] - trace[0]) / (peak_idx / sampling_rate)
    return rise_rate

# Function to calculate decay rate (from peak to end)
def calculate_decay_rate(trace, sampling_rate):
    peak_idx = np.argmax(trace)
    decay_rate = (trace[peak_idx] - trace[-1]) / ((len(trace) - peak_idx) / sampling_rate)
    return decay_rate

# Function to calculate area under the curve (AUC)
def calculate_auc(trace, sampling_rate):
    auc = np.trapz(trace, dx=1/sampling_rate)
    return auc

# Extract metrics from traces
def extract_metrics(traces, sampling_rate):
    fwhm_list = []
    peak_amplitude_list = []
    rise_rate_list = []
    decay_rate_list = []
    auc_list = []
    
    for trace in traces:
        fwhm_list.append(calculate_fwhm(trace, sampling_rate))
        peak_amplitude_list.append(calculate_peak_amplitude(trace))
        rise_rate_list.append(calculate_rise_rate(trace, sampling_rate))
        decay_rate_list.append(calculate_decay_rate(trace, sampling_rate))
        auc_list.append(calculate_auc(trace, sampling_rate))
    
    return {
        'FWHM': fwhm_list,
        'Peak Amplitude': peak_amplitude_list,
        'Rise Rate': rise_rate_list,
        'Decay Rate': decay_rate_list,
        'AUC': auc_list
    }

# Perform non-parametric statistical comparisons
def perform_statistical_tests(metrics_df, metric, group1, group2):
    gr1=metrics_df[metric]
    group1_values = gr1[group1]
    group2_values = gr1[group2]
    stat, p_value = stats.mannwhitneyu(group1_values, group2_values, alternative='two-sided')
    return p_value

# Function to plot metrics with significance annotations
def plot_metrics(metrics_df, metric, durations, regions, p_values, colors):
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Duration', y=metric, hue='Region', data=metrics_df, split=True, inner='quartile', palette=colors)
    plt.title(f'{metric} Comparison by Duration and Region')
    
    for i, duration in enumerate(durations):
        for region in regions:
            plt.scatter(x=[i]*len(metrics_df[(metrics_df['Duration'] == duration) & (metrics_df['Region'] == region)]),
                        y=metrics_df[(metrics_df['Duration'] == duration) & (metrics_df['Region'] == region)][metric],
                        color='black', s=50, zorder=2)
    
    # Annotate significance
    for p_value_dict in p_values:
        if p_value_dict['Metric'] == metric and p_value_dict['p-value'] < 0.05:
            group1_parts = p_value_dict['Group 1'].split()
            group2_parts = p_value_dict['Group 2'].split()
            duration1 = int(group1_parts[-1])
            duration2 = int(group2_parts[-1])
            region1 = group1_parts[0]
            region2 = group2_parts[0]
            y_max = max(metrics_df[metric])
            if region1 == region2:
                plt.text((durations.index(duration1) + durations.index(duration2)) / 2, y_max, '*', fontsize=20, ha='center')
    
    plt.show()


def plot_metrics_within(metrics_df, metric, durations, regions, p_values, colors):
    plt.figure(figsize=(12, 8))
    
    # Create a new column to differentiate the groups
    metrics_df['Group'] = metrics_df['Session'] + " " + metrics_df['Region']
    
    # Define the order for the groups
    group_order = [
        'Before_Cocaine DS', 'After_Cocaine DS',
        'Before_Cocaine VS', 'After_Cocaine VS',

    ]
    
    # Convert the 'Group' column to a categorical type with the specified order
    metrics_df['Group'] = pd.Categorical(metrics_df['Group'], categories=group_order, ordered=True)
    
    palette = {
        'Before_Cocaine VS': 'blue',
        'After_Cocaine VS': 'green',
        'Before_Cocaine DS': 'purple',
        'After_Cocaine DS': 'orange'
    }
        
    # Adjust the violin plot to use 'Group' as hue
    sns.violinplot(x='Duration', y=metric, hue='Group', data=metrics_df, split=True, inner='quartile', palette=palette, hue_order=group_order)
    plt.title(f'{metric} Comparison by Duration and Region')
    
    
    # Annotate significance
    for p_value_dict in p_values:
        if p_value_dict['Metric'] == metric and p_value_dict['p-value'] < 0.05:
            group1_parts = p_value_dict['Group 1'].rsplit(' ', 2)
            group2_parts = p_value_dict['Group 2'].rsplit(' ', 2)
            duration1 = int(group1_parts[-1])
            duration2 = int(group2_parts[-1])
            region1 = group1_parts[0]
            region2 = group2_parts[0]
            y_max = max(metrics_df[metric])
            if region1 == region2:
                plt.text((durations.index(duration1) + durations.index(duration2)) / 2, y_max, '*', fontsize=20, ha='center')
    
    plt.savefig('results/plots/'+metric+ 'stat_comparison'+'.pdf')


# normalisaion of the FOM traces by the z score, based on the baseline
def normalize_trace(trace, norm_end_index):
    baseline_mean = np.mean(trace[:norm_end_index])
    baseline_std = np.std(trace[:norm_end_index])
    normalized_trace = trace 
    return normalized_trace

# extract and normalize traces for given conditions
def get_normalized_traces(df, region_id, duration, start_index, end_index, norm_end_index):
    region_df = df[(df['region_id'] == region_id) & (df['duration'] == duration)]
    valid_traces = region_df['data'].apply(lambda x: x[start_index:end_index])
    normalized_traces = valid_traces.apply(lambda x: normalize_trace(x, norm_end_index))
    return normalized_traces

#metric extraction

# Function to fit exponential decay and extract decay constant (Tau)
def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def fit_exponential_decay(trace, sampling_rate):
    x_data = np.arange(len(trace)) / sampling_rate
    try:
        popt, _ = curve_fit(exponential_decay, x_data, trace, p0=(trace[0], 1, trace[-1]), maxfev=4000)
        return 1 / popt[1]
    except RuntimeError as e:
        print(f"Error fitting exponential decay: {e}")
        return np.nan


# Function to extract additional metrics
def extract_additional_metrics(traces, sampling_rate):
    metrics = {
        'TTP': [],
        'Half-Decay Time': [],
        'Max Slope': [],
        #'Tau': [],
        'Rise Time': [],
        'Decay Time': [],
        'FDHM': [],
        'Skewness': [],
        'Kurtosis': [],
        'Baseline Drift': [],
        'Noise Level': [],
        'Peak-to-Baseline Ratio': []
    }
    
    for trace in traces:
        peak_idx = np.argmax(trace)
        peak_amplitude = trace[peak_idx]
        ttp = peak_idx / sampling_rate
        half_max = peak_amplitude / 2
        above_half_max = np.where(trace >= half_max)[0]
        if len(above_half_max) > 1:
            half_decay_time = (above_half_max[-1] - peak_idx) / sampling_rate
            fdhm = (above_half_max[-1] - above_half_max[0]) / sampling_rate
        else:
            half_decay_time = np.nan
            fdhm = np.nan
        max_slope = np.max(np.diff(trace[:peak_idx -1]) * sampling_rate)
        rise_time = (np.where(trace >= 0.9 * peak_amplitude)[0][0] - np.where(trace >= 0.1 * peak_amplitude)[0][0]) / sampling_rate
        decay_time = (np.where(trace <= 0.1 * peak_amplitude)[0][-1] - np.where(trace <= 0.9 * peak_amplitude)[0][-1]) / sampling_rate
        #tau = fit_exponential_decay(trace[peak_idx:], sampling_rate)
        trace_skewness = skew(trace)
        trace_kurtosis = kurtosis(trace)
        baseline_drift = trace[-1] - trace[0]
        noise_level = np.std(trace[:10])
        peak_to_baseline_ratio = peak_amplitude / trace[0]

        metrics['TTP'].append(ttp)
        metrics['Half-Decay Time'].append(half_decay_time)
        metrics['Max Slope'].append(max_slope)
        #metrics['Tau'].append(tau)
        metrics['Rise Time'].append(rise_time)
        metrics['Decay Time'].append(decay_time)
        metrics['FDHM'].append(fdhm)
        metrics['Skewness'].append(trace_skewness)
        metrics['Kurtosis'].append(trace_kurtosis)
        metrics['Baseline Drift'].append(baseline_drift)
        metrics['Noise Level'].append(noise_level)
        metrics['Peak-to-Baseline Ratio'].append(peak_to_baseline_ratio)
    
    return metrics



def get_valid_normalized_traces_df(df, session_ids, region_id, duration, start_index, end_index, norm_end_index):
    filtered_df = df[df['session_id'].isin(session_ids)]
    region_df = filtered_df[(filtered_df['region_id'] == region_id) & (filtered_df['duration'] == duration)]
    valid_traces = region_df['data'].apply(lambda x: x[start_index:end_index])
    normalized_traces = valid_traces.apply(lambda x: normalize_trace(x, norm_end_index))
    return pd.DataFrame({'session_id': region_df['session_id'], 'duration': region_df['duration'], 'region_id': region_df['region_id'], 'data': valid_traces, 'normalized_data': normalized_traces})



################################## normalisation 

def remove_trend_polyfit(data, degree=2):
    x = np.arange(len(data))
    p = np.polyfit(x, data, degree)
    trend = np.polyval(p, x)
    detrended_data = data - trend
    return detrended_data

def robust_zscore(data):
    median = np.median(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    return (data - median) / iqr

def resample_signal(data, original_rate, target_rate):
    num_samples = int(len(data) * (target_rate / original_rate))
    resampled_data = resample(data, num_samples)
    return resampled_data

# High-pass filter to remove very slow frequencies (e.g., below 0.05 Hz)
def high_pass_filter(data, fs, cutoff=0.05):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Low-pass filter to focus on slow frequencies (e.g., below 5 Hz)
def low_pass_filter(data, fs, cutoff=5.0):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


# Function to extract unique identifier from the "file" field
def get_animal_id(file_name):
    return file_name[:9]

# Function to compute the power spectrum using Welch's method and convert it to dB
def compute_power_spectrum_dB(data, fs, nperseg=4096, noverlap=None, max_freq=30):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # If noverlap is not provided, default to 75% of nperseg
    if noverlap is None:
        noverlap = nperseg * 3 // 4
    
    # Compute the power spectrum using Welch's method
    freqs, power = welch(data, fs, nperseg=nperseg, noverlap=noverlap)
    
    # Convert power to decibels (dB)
    power_dB = np.log(power)
    
    # Focus on the very low frequencies, up to max_freq Hz
    mask = freqs <= max_freq
    return np.log(freqs[mask]), power[mask]

# Function to extract unique identifier from the "file" field
def get_animal_id(file_name):
    return file_name[:9]


def simple_deconvolution(signal, template_signal):
    # Perform convolution-based template matching
    template = template_signal / np.sqrt(np.sum(template_signal**2))

    deconvolved_signal = convolve(signal, template[::-1], mode='same')
    scaling_factor = np.sqrt(np.sum(signal**2) / np.sum(deconvolved_signal**2))
    deconvolved_signal *= scaling_factor
    return deconvolved_signal

def evaluate_deconvolution(original_signal, deconvolved_signal):
    # Signal-to-Noise Ratio (SNR)
    snr = np.mean(deconvolved_signal) / np.std(deconvolved_signal)
    
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(mean_squared_error(original_signal, deconvolved_signal))
    
    # Correlation Coefficient
    correlation = np.corrcoef(original_signal, deconvolved_signal)[0, 1]

    return snr, rmse, correlation

def load_template(template_file_path, dur):
    if not os.path.exists(template_file_path):
        logging.error(f"File not found: {template_file_path}")
        raise FileNotFoundError(f"File not found: {template_file_path}")
    template_ = pd.read_csv(template_file_path)
    template = template_[dur].values
    logging.info(f"Loaded template from {template_file_path}")
    return np.array(template)

def richardson_lucy(signal, template, iterations=10):
    template = template / np.sqrt(np.sum(template**2))

    template_flip = template[::-1]
    deconvolved = np.ones_like(signal)
    
    for i in range(iterations):
        relative_blur = signal / (convolve(deconvolved, template, 'same') + 1e-5)
        deconvolved *= convolve(relative_blur, template_flip, 'same')
    
    return deconvolved