import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis
import seaborn as sns 
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
