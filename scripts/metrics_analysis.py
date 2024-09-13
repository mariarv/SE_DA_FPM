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