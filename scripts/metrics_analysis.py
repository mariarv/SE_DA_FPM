import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis, lognorm, norm
import seaborn as sns 
from scipy.signal import welch, butter, filtfilt, resample
import pickle
from scipy.signal import convolve, find_peaks
from sklearn.metrics import mean_squared_error
import logging
import os 
import matplotlib.cm as cm
import math

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
    trace=trace-np.mean(trace[0:10])
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
    trace=trace-np.mean(trace[0:10])
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
    #region_df = df[(df['region_id'] == region_id) & (df['duration'] == duration)]
    region_df = df[(df['duration'] == duration)]
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
        #'Rise Time': [],
        #'Decay Time': [],
        'FDHM': [],
        'Skewness': [],
        'Kurtosis': [],
        'Baseline Drift': [],
        'Noise Level': [],
        'Peak-to-Baseline Ratio': []
    }
    
    for trace in traces:
        if isinstance(trace, list): 
            trace = np.array(trace)
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
        #rise_time = (np.where(trace >= 0.9 * peak_amplitude)[0][0] - np.where(trace >= 0.1 * peak_amplitude)[0][0]) / sampling_rate
        below_10_percent, below_90_percent = np.where(trace <= 0.1 * peak_amplitude)[0], np.where(trace <= 0.9 * peak_amplitude)[0]
        decay_time = (below_10_percent[-1] - below_90_percent[-1]) / sampling_rate if below_10_percent.size > 0 and below_90_percent.size > 0 else math.nan
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
        #metrics['Rise Time'].append(rise_time)
        #metrics['Decay Time'].append(decay_time)
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
    #mean=np.mean(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    return (data - median)/iqr

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



#################### single waveform analysis: 

def extract_waveforms(signal, spike_times, original_rate, pre_spike_ms=400, max_duration_sec=1.2):
    """
    Extracts baseline-normalized waveforms from the signal around each detected spike.
    
    Parameters:
    - signal: The original signal array.
    - spike_times: Indices of detected spikes.
    - original_rate: The sampling rate of the signal.
    - pre_spike_ms: Time to capture before the spike in milliseconds.
    - max_duration_sec: Maximum duration of the waveform after the spike in seconds.
    
    Returns:
    - waveforms: A list of baseline-normalized extracted waveforms.
    """
    waveforms = []
    pre_spike_samples = int(pre_spike_ms / 1000 * original_rate)
    max_duration_samples = int(max_duration_sec * original_rate)
    min_distance_samples = int(1.5* original_rate)  # Convert ms to samples
    
    # Filter spikes to ensure that no two spikes are closer than min_distance_ms
    filtered_spike_times = []
    
    for i, spike in enumerate(spike_times):
        if i == 0:
            # Add the first spike by default
            filtered_spike_times.append(spike)
        else:
            # Check if current spike is far enough from the previous one
            if spike - filtered_spike_times[-1] >= min_distance_samples:
                # Check if there is a next spike and if it is far enough from the current one
                if i < len(spike_times) - 1:
                    next_spike = spike_times[i + 1]
                    if next_spike - spike >= min_distance_samples:
                        filtered_spike_times.append(spike)


    for i, spike in enumerate(filtered_spike_times):
        # Determine end time based on the next spike or the max duration (whichever is smaller)
        if i < len(filtered_spike_times) - 1:
            next_spike = filtered_spike_times[i + 1]
            end_time = min(next_spike - pre_spike_samples, spike + max_duration_samples)
        else:
            end_time = min(len(signal), spike + max_duration_samples)
        
        # Ensure end_time is valid and extract waveform
        if end_time > spike:
            start_time = max(0, spike - pre_spike_samples)
            waveform = signal[start_time:end_time]
            normalized_waveform = waveform

            # Baseline normalization: subtract the average of the first 500 ms (pre_spike_samples)
            baseline = np.nanmean(normalized_waveform[0:0+10])
            normalized_waveform = normalized_waveform - 0  # Subtract baseline to bring the start to 0
            if len(normalized_waveform)>int(0.8 * original_rate): 
                waveforms.append(normalized_waveform)
    
    return waveforms

def extract_valid_trace(trace, sampling_rate):
    # Find the index of the peak in the trace
    peak_index = np.argmax(trace)  # Adjust if using different peak-finding logic
    
    # Calculate the start and end indices
    start_index = max(peak_index - int(0.4 * sampling_rate), 0)  # Ensure start_index is not negative
    end_index = min(start_index + int(2.6 * sampling_rate), len(trace))  # Ensure end_index is within trace length
    
    # Extract the valid portion of the trace (1 second window)
    valid_trace = trace[start_index:end_index]
    
    return valid_trace

def calculate_spike_statistics(waveforms, original_rate):
    """
    Calculate statistics such as amplitude, decay rates, and widths for the spikes.
    
    Parameters:
    - waveforms: A list of extracted waveforms.
    - original_rate: The sampling rate of the signal.
    
    Returns:
    - stats: A dictionary with spike statistics (amplitude, decay rate, width).
    """
    amplitudes = []
    decay_rates = []
    widths = []
    
    for waveform in waveforms:
        # Amplitude is the max value of the waveform
        amplitude = np.max(waveform)
        amplitudes.append(amplitude)
        
        # Decay rate: estimate using the peak and the decay slope after peak
        peak_index = np.argmax(waveform)
        if len(waveform) > peak_index + 1:
            decay_slope = -(waveform[peak_index] - waveform[-1]) / (len(waveform) - peak_index)
        else:
            decay_slope = 0
        decay_rates.append(decay_slope)
        
        # Width: Full Width at Half Maximum (FWHM)
        half_amplitude = amplitude / 2
        left_idx = np.where(waveform[:peak_index] <= half_amplitude)[0]
        right_idx = np.where(waveform[peak_index:] <= half_amplitude)[0]
        if len(left_idx) > 0 and len(right_idx) > 0:
            width = (right_idx[0] + peak_index) - left_idx[-1]
        else:
            width = 0
        widths.append(width / original_rate)  # Convert to seconds
    
    stats = {
        'amplitude': amplitudes,
        'decay_rate': decay_rates,
        'width': widths
    }
    
    return stats

def plot_waveforms(waveforms, original_rate):
    """
    Plot all baseline-normalized waveforms in one plot to visualize their shapes.
    
    Parameters:
    - waveforms: A list of extracted and baseline-normalized waveforms.
    - original_rate: The sampling rate of the signal.
    """
    plt.figure(figsize=(10, 6))
    for i, waveform in enumerate(waveforms):
        time_vector = np.arange(len(waveform)) / original_rate
        plt.plot(time_vector, waveform, label=f'Waveform {i + 1}', alpha=0.5)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.title('Extracted Baseline-Normalized Spike Waveforms')
    plt.show()

def plot_spike_statistics(stats):
    """
    Plot swarm plots for spike statistics: amplitude, decay rates, and widths.
    
    Parameters:
    - stats: A dictionary with spike statistics (amplitude, decay rate, width).
    """
    plt.figure(figsize=(18, 6))
    
    # Amplitude
    plt.subplot(1, 3, 1)
    sns.swarmplot(data=stats['amplitude'], color='blue')
    plt.title('Spike Amplitudes')
    plt.ylabel('Amplitude')

    # Decay Rates
    plt.subplot(1, 3, 2)
    sns.swarmplot(data=stats['decay_rate'], color='green')
    plt.title('Spike Decay Rates')
    plt.ylabel('Decay Rate')

    # Widths
    plt.subplot(1, 3, 3)
    sns.swarmplot(data=stats['width'], color='red')
    plt.title('Spike Widths (FWHM)')
    plt.ylabel('Width (s)')

    plt.tight_layout()
    plt.show()


def plot_waveforms_with_color(waveforms, widths, original_rate):
    """
    Plot all baseline-normalized waveforms, color-coded based on their width.
    
    Parameters:
    - waveforms: A list of extracted and baseline-normalized waveforms.
    - widths: A list of widths (FWHM) for each waveform.
    - original_rate: The sampling rate of the signal.
    """
    # Normalize widths to a range [0, 1] to map them to a colormap
    normalized_widths = (widths - np.min(widths)) / (np.max(widths) - np.min(widths))
    
    # Define a colormap (e.g., from light blue to dark blue)
    colormap = plt.get_cmap('cool')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, waveform in enumerate(waveforms):
        time_vector = np.arange(len(waveform)) / original_rate
        color = colormap(normalized_widths[i])  # Assign color based on width
        plt.plot(time_vector, waveform, color=color, label=f'Waveform {i + 1}', alpha=0.7)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.title('Extracted Baseline-Normalized Spike Waveforms (Color-coded by Width)')
    
    # Create a colorbar to show the mapping from width to color
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=np.min(widths), vmax=np.max(widths)))
    sm.set_array([])  # Only needed for the colorbar
    cbar = fig.colorbar(sm, ax=ax)  # Link the colorbar to the current axis
    cbar.set_label('Spike Width (s)', rotation=270, labelpad=15)
    
    plt.show()


def calculate_spike_statistics_with_next_spike(waveforms, spike_times, original_rate):
    """
    Calculate statistics such as amplitude, decay rates, widths, and time to next spike for the spikes.
    
    Parameters:
    - waveforms: A list of extracted waveforms.
    - spike_times: A list of spike times.
    - original_rate: The sampling rate of the signal.
    
    Returns:
    - stats: A dictionary with spike statistics (amplitude, decay rate, width, time to next spike).
    """
    amplitudes = []
    decay_rates = []
    widths = []
    time_to_next_spike = []

    for i, waveform in enumerate(waveforms):
        # Amplitude is the max value of the waveform
        amplitude = np.max(waveform)
        amplitudes.append(amplitude)
        
        # Decay rate: estimate using the peak and the decay slope after peak
        peak_index = np.argmax(waveform)
        if len(waveform) > peak_index + 1:
            decay_slope = -(waveform[peak_index] - waveform[-1]) / (len(waveform) - peak_index)
        else:
            decay_slope = 0
        decay_rates.append(decay_slope)
        
        # Width: Full Width at Half Maximum (FWHM)
        half_amplitude = amplitude / 2
        left_idx = np.where(waveform[:peak_index] <= half_amplitude)[0]
        right_idx = np.where(waveform[peak_index:] <= half_amplitude)[0]
        if len(left_idx) > 0 and len(right_idx) > 0:
            width = (right_idx[0] + peak_index) - left_idx[-1]
        else:
            width = 0
        widths.append(width / original_rate)  # Convert to seconds

        # Time to next spike
        if i < len(spike_times) - 1:
            next_spike_time = spike_times[i + 1]
            time_diff = (next_spike_time - spike_times[i]) / original_rate  # Time in seconds
        else:
            time_diff = None  # No next spike
        time_to_next_spike.append(time_diff)
    
    stats = {
        'amplitude': amplitudes,
        'decay_rate': decay_rates,
        'width': widths,
        'time_to_next_spike': time_to_next_spike
    }
    
    return stats

def plot_time_to_next_spike_vs_amplitude(stats):
    """
    Plot time to next spike vs amplitude of spike.
    
    Parameters:
    - stats: A dictionary containing spike statistics, including time to next spike and amplitude.
    """
    time_to_next_spike = np.array(stats['time_to_next_spike'])
    amplitudes = np.array(stats['amplitude'])
    
    # Filter out None values from time_to_next_spike
    valid_indices = np.where(time_to_next_spike != None)[0]
    valid_time_to_next_spike = time_to_next_spike[valid_indices]
    valid_amplitudes = amplitudes[valid_indices]

    plt.figure(figsize=(8, 6))
    plt.scatter(valid_amplitudes, valid_time_to_next_spike, c='blue', alpha=0.7)
    plt.xlabel('Spike Amplitude')
    plt.ylabel('Time to Next Spike (s)')
    plt.title('Time to Next Spike vs Spike Amplitude')
    plt.show()

def plot_raster(spike_trains, dt):
    """
    spike_trains: A 2D numpy array where each row represents the spike train of a neuron.
                  Each element is binary (1 for spike, 0 for no spike) or a list of spike timestamps.
    dt: Time step used for the simulation in ms or time units.
    """
    num_neurons = len(spike_trains)
    
    # Create a new figure for the raster plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Loop over each neuron and plot its spike times
    for neuron_idx, spike_train in enumerate(spike_trains):
        spike_times = np.where(spike_train > 0)[0] * dt  # Extract spike times based on non-zero entries
        ax.scatter(spike_times, np.ones_like(spike_times) * neuron_idx, s=2, color="black")

    ax.set_title('Raster Plot of Neuronal Spiking Activity')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.set_ylim(-0.5, num_neurons - 0.5)
######################################################################################################
#custom peak search 

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

def scale_to_match_combined(normalized_combined_pdf, scaled_component_pdf, burst_isis,x_values, region='burst'):
    if region == 'burst':
        component_peak_index = np.argmax(scaled_component_pdf[x_values < 0.05])  # Burst peak index
    else:
        component_peak_index = np.argmax(scaled_component_pdf[x_values >=  np.median(burst_isis)])  # Tonic peak index

    component_peak_value = scaled_component_pdf[component_peak_index]
    combined_peak_value = normalized_combined_pdf[component_peak_index]
    scaling_factor = combined_peak_value / component_peak_value if component_peak_value > 0 else 1.0
    scaled_component_pdf = scaled_component_pdf * scaling_factor
    return scaled_component_pdf

# Function to extract parameters and combined distribution for each cell
def extract_combined_distribution_for_cell(neuron_isis, x_values):
    burst_isis = neuron_isis[neuron_isis < 0.05]
    tonic_isis = neuron_isis[neuron_isis >= np.median(burst_isis)]

    burst_shape, _, burst_scale = lognorm.fit(burst_isis, floc=0)
    tonic_mean,loc, tonic_std = lognorm.fit(tonic_isis)

    burst_pdf = lognorm.pdf(x_values, burst_shape, 0, burst_scale)
    tonic_pdf = lognorm.pdf(x_values, tonic_mean,loc, tonic_std)

    scaled_burst_pdf = burst_pdf.copy()
    scaled_tonic_pdf = tonic_pdf.copy()

    combined_pdf = scaled_burst_pdf + scaled_tonic_pdf
    normalized_combined_pdf = combined_pdf / np.trapz(combined_pdf, x_values)  # Normalize combined PDF

    # Scale the burst and tonic components to match the combined PDF
    scaled_burst_pdf_matched = scale_to_match_combined(normalized_combined_pdf, scaled_burst_pdf,burst_isis, x_values, region='burst')
    scaled_tonic_pdf_matched = scale_to_match_combined(normalized_combined_pdf, scaled_tonic_pdf, burst_isis,x_values, region='tonic')

    # Return the normalized combined PDF and the individual components
    return normalized_combined_pdf, scaled_burst_pdf_matched, scaled_tonic_pdf_matched

# Function to sample ISIs from a given PDF
def sample_spiking_for_duration(pdf, x_values, duration=60):
    """
    Samples inter-spike intervals (ISIs) based on the provided PDF until the total time reaches the specified duration.
    Args:
        pdf (np.array): The probability density function to sample from.
        x_values (np.array): The x-axis values corresponding to the PDF.
        duration (float): The desired duration in seconds.

    Returns:
        spike_times (np.array): Array of cumulative spike times up to the specified duration.
    """
    pdf_normalized = pdf / np.sum(pdf)  # Normalize the PDF to get a proper probability distribution
    spike_times = []  # List to accumulate spike times
    total_time = 0  # Track the cumulative time

    while total_time < duration:
        # Sample a single ISI based on the PDF
        sampled_isi_index = np.random.choice(len(x_values), p=pdf_normalized)
        sampled_isi = x_values[sampled_isi_index]

        # Update cumulative time and add the spike time
        total_time += sampled_isi
        if total_time <= duration:  # Only add the spike if it does not exceed the duration
            spike_times.append(total_time)

    return np.array(spike_times)
def create_hankel_matrix(signal, window_size):
    return hankel(signal[:window_size], signal[window_size-200:])

def create_sliding_windows(signal, window_size, step_size):
    num_windows = (len(signal) - window_size) // step_size + 1
    return np.array([signal[i*step_size : i*step_size + window_size] for i in range(num_windows)])