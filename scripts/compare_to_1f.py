import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, resample
from collections import defaultdict
from scipy.signal import detrend
from scipy.stats import sem, t
import metrics_analysis as m_a
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import umap
from scipy.optimize import curve_fit
import seaborn as sns
from scipy.stats import wilcoxon

# Constants
ORIGINAL_RATE = 1017.252625
TARGET_RATE = 1000
SEGMENT_DURATION = 3 * 60  # 3 minutes in seconds

# Paths to the pickle files created previously
pickle_file_path_vs = 'df_combined_vs_all_drugs.pkl'
pickle_file_path_ds = 'df_combined_ds_all_drugs.pkl'

def one_over_f(freqs, A, n):
    return A * np.exp(-n * freqs)
def normalize_curve(psd):
    # Remove NaN and Inf values
    clean_psd = psd[np.isfinite(psd)]
    
    if len(clean_psd) == 0:
        raise ValueError("All values in the PSD are NaN or Inf, cannot normalize.")
    
    # Normalize the PSD by the maximum finite value
    return psd / np.max(clean_psd)
# Function to perform the 1/f fitting
def fit_one_over_f(freqs, psd):
    # Remove non-finite values (NaN, Inf) from both freqs and psd
    mask = np.isfinite(freqs) & np.isfinite(psd)
    clean_freqs = freqs[mask]
    clean_psd = psd[mask]
    threshold = 1e-10
    clean_freqs = clean_freqs[clean_freqs > threshold]
    clean_psd = clean_psd[clean_psd > threshold]

    # Ensure shapes match after filtering
    clean_freqs = clean_freqs[:len(clean_psd)]
    clean_psd = clean_psd[:len(clean_freqs)]
    if len(clean_freqs) == 0 or len(clean_psd) == 0:
        raise ValueError("No valid data points for fitting 1/f model.")

    # Fit the 1/f model
    try:
        popt, _ = curve_fit(one_over_f, clean_freqs, clean_psd, bounds=(0, [np.inf, 3]))
    except ValueError as e:
        print(f"Error during curve fitting: {e}")
        raise

    return popt  # A, n


# Function to segment the data into three 3-minute segments
def segment_data(data, fs, condition_name):
    """
    Segments the data based on the condition.
    
    Parameters:
    - data: The time series data to be segmented.
    - fs: Sampling frequency.
    - condition_name: The condition name ('base_after', 'opto_drug', etc.).
    
    Returns:
    A dictionary containing the segmented data.
    """
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


# Function to process the base_before, opto_drug, and base_after traces
def process_trace(data, fs, trace_type):
    # Resample to target rate
    resampled_data = m_a.resample_signal(data, ORIGINAL_RATE, TARGET_RATE)
    
    if trace_type == 'base_before':
        processed_data = segment_data(resampled_data, fs,trace_type)['first_3min']  # First 3 minutes
        #processed_data = m_a.high_pass_filter(processed_data, fs)
        #processed_data = m_a.robust_zscore(processed_data)
    elif trace_type == 'opto_drug':
        processed_data = segment_data(resampled_data, fs,trace_type)['first_3min']  # First 3 minutes
        #processed_data = m_a.high_pass_filter(processed_data, fs)
        #processed_data = m_a.robust_zscore(processed_data)
    elif trace_type == 'base_after':
        processed_data = segment_data(resampled_data, fs,trace_type)
        for key in processed_data:
            processed_data[key] = processed_data[key]
            #processed_data[key] = m_a.high_pass_filter(processed_data[key], fs)
            #processed_data[key] = m_a.remove_trend_polyfit(processed_data[key])
            #processed_data[key] = m_a.robust_zscore(processed_data[key])
    else:
        processed_data = None

    return processed_data

###########  Confidence Envelopes
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

def plot_spectrum_with_confidence_envelopes(segments_vs, ids_vs, segments_ds, ids_ds, fs, segment_name, drug_name):
    plt.figure(figsize=(14, 10), facecolor='black')

    # Get current axes and set its background color to black
    ax = plt.gca()
    ax.set_facecolor('black')

    # VS: Calculate and plot mean spectrum with confidence envelopes
    freqs_vs, mean_psd_vs, lower_vs, upper_vs = compute_spectrum_with_confidence(segments_vs, fs)
    plt.plot(freqs_vs, mean_psd_vs, color='yellow', label='VS Mean Spectrum')
    plt.fill_between(freqs_vs, lower_vs, upper_vs, color='yellow', alpha=0.2, label='VS Confidence Envelope')

    # Set title and labels with white color
    plt.title(f'VS {segment_name.replace("_", " ").capitalize()} - Spectrum with Confidence ({drug_name})', color='white')
    plt.xlabel('Frequency (Hz)', color='white')
    plt.ylabel('Power (dB)', color='white')

    # Customize tick parameters to be white
    ax.tick_params(axis='both', colors='white')

    # Set spine colors to white
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Customize the legend
    legend = plt.legend(loc='upper right')
    # Set legend text color to white
    for text in legend.get_texts():
        text.set_color('white')
    # Set legend background to black
    legend.get_frame().set_facecolor('black')
    # Set legend edge color to white
    legend.get_frame().set_edgecolor('white')

    # Optional: Adjust grid lines if you have them
    # ax.grid(True, color='gray')
    
    #freqs_ds, mean_psd_ds, lower_ds, upper_ds = compute_spectrum_with_confidence(segments_ds, fs)
    #plt.plot(freqs_ds, mean_psd_ds, color='orange', label='DS Mean Spectrum')
    #plt.title(f'DS {segment_name.replace("_", " ").capitalize()} - Spectrum with Confidence ({drug_name})')
    ##plt.fill_between(freqs_ds, lower_ds, upper_ds, color='orange', alpha=0.2, label='DS Confidence Envelope')
    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Power (dB)')
    #plt.legend(loc='upper right')

    #freqs_ds_base, mean_psd_ds, lower_ds, upper_ds = compute_spectrum_with_confidence(segments_ds, fs)

    
    plt.tight_layout()

    plt.show()

def combine_and_plot_spectra_with_envelopes(df_vs, df_ds, fs, condition_name):
    combined_segments_vs = []
    combined_segments_ds = []

    # Combine VS data across all drugs
    for index, row in df_vs.iterrows():
        if len(row[condition_name]) > 0:
            processed_data = process_trace(np.array(row[condition_name]), fs, condition_name)
            combined_segments_vs.append(processed_data)

    # Combine DS data across all drugs
    for index, row in df_ds.iterrows():
        if len(row[condition_name]) > 0:
            processed_data = process_trace(np.array(row[condition_name]), fs, condition_name)
            combined_segments_ds.append(processed_data)

    # Plot combined spectra with confidence envelopes
    plot_spectrum_with_confidence_envelopes(combined_segments_vs, None, combined_segments_ds, None, fs, condition_name, "Combined")

########

# Function to plot traces and power spectra for each segment
def plot_segment_analysis(segments_vs, ids_vs, segments_ds, ids_ds, fs, segment_name, drug_name):
    plt.figure(figsize=(14, 10))
    
    # Define color palette for consistent coloring across animals
    colors = plt.get_cmap('tab10', max(len(segments_vs), len(segments_ds)))
    
    # Top-left: VS filtered traces
    plt.subplot(2, 2, 1)
    for i, segment in enumerate(segments_vs):
        filtered_segment = m_a.low_pass_filter(segment, fs)  # Apply low-pass filtering for opto_drug and base_after
        plt.plot(np.arange(len(filtered_segment)) / fs, filtered_segment, color=colors(i), label=f'Animal {ids_vs[i]}')
    plt.title(f'VS {segment_name.replace("_", " ").capitalize()} - Filtered Traces ({drug_name})')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    #plt.legend(loc='upper right')
    
    # Top-right: DS filtered traces
    plt.subplot(2, 2, 2)
    for i, segment in enumerate(segments_ds):
        filtered_segment = m_a.low_pass_filter(segment, fs)  # Apply low-pass filtering for opto_drug and base_after
        plt.plot(np.arange(len(filtered_segment)) / fs, filtered_segment, color=colors(i), label=f'Animal {ids_ds[i]}')
    plt.title(f'DS {segment_name.replace("_", " ").capitalize()} - Filtered Traces ({drug_name})')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    #plt.legend(loc='upper right')
    
    # Bottom-left: VS power spectra in dB
    plt.subplot(2, 2, 3)
    for i, segment in enumerate(segments_vs):
        freqs, power_dB =  m_a.compute_power_spectrum_dB(segment, fs)
        plt.plot(freqs, power_dB, color=colors(i), label=f'Animal {ids_vs[i]}')
    plt.title(f'VS {segment_name.replace("_", " ").capitalize()} - Power Spectrum (dB) ({drug_name})')
    plt.xlabel('log(Frequency (Hz)) ')
    plt.ylabel('Power')
    plt.legend(loc='upper right')
    
    # Bottom-right: DS power spectra in dB
    plt.subplot(2, 2, 4)
    for i, segment in enumerate(segments_ds):
        freqs, power_dB =  m_a.compute_power_spectrum_dB(segment, fs)
        plt.plot(freqs, power_dB, color=colors(i), label=f'Animal {ids_ds[i]}')
    plt.title(f'DS {segment_name.replace("_", " ").capitalize()} - Power Spectrum (dB) ({drug_name})')
    plt.xlabel('log(Frequency (Hz))')
    plt.ylabel('Power')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'results/{segment_name.replace("_", " ").capitalize()} - Power Spectrum (dB) ({drug_name}).pdf')
    plt.show()
def one_over_f(freqs, A, alpha):
    return A / freqs**alpha

def fit_one_over_f(freqs, psd):
    # Remove zero and negative frequencies
    mask = (freqs > 0) & (psd > 0) & np.isfinite(psd)
    clean_freqs = freqs[mask]
    clean_psd = psd[mask]

    log_freqs = np.log(clean_freqs)
    log_psd = np.log(clean_psd)

    # Fit linear model: log_psd = log_A - alpha * log_freqs
    def linear_model(log_f, log_A, alpha):
        return log_A - alpha * log_f

    popt, pcov = curve_fit(linear_model, log_freqs, log_psd)
    log_A, alpha = popt
    A = np.exp(log_A)

    # Compute R²
    residuals = log_psd - linear_model(log_freqs, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_psd - np.mean(log_psd))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return A, alpha, r_squared

def plot_individual_spectra_second_segment(df_vs, df_ds, fs, drug):
    """
    Plot spectra for the second 3-minute segment of 'base_after' condition for the given drug,
    with one subplot per individual. Separate plots for VS and DS conditions.
    """
    # Extract individual IDs (assumed to be the first 9 characters of 'file')
    individuals_vs = df_vs['file'].str[:9].unique()
    individuals_ds = df_ds['file'].str[:9].unique()

    # Define color for the drug
    color = 'green'

    # VS Plot
    num_individuals_vs = len(individuals_vs)
    fig_vs, axes_vs = plt.subplots(nrows=num_individuals_vs, ncols=1, figsize=(10, 4 * num_individuals_vs), facecolor='black')

    for i, individual in enumerate(individuals_vs):
        ax = axes_vs[i] if num_individuals_vs > 1 else axes_vs

        df_vs_individual = df_vs[df_vs['file'].str.startswith(individual)]
        df_vs_drug = df_vs_individual[df_vs_individual['drug'] == drug]

        combined_segments_vs_after = []
        combined_segments_vs_before = []

        for index, row in df_vs_drug.iterrows():
            if len(row['base_after']) > 0:
                processed_data = process_trace(np.array(row['base_after']), fs, 'base_after')
                combined_segments_vs_after.append(processed_data['second_3min'])
            if len(row['base_before']) > 0:
                processed_data = process_trace(np.array(row['base_before']), fs, 'base_before')
                combined_segments_vs_before.append(processed_data)

        # Compute and plot spectra
        if combined_segments_vs_after:
            freqs_vs_after, mean_psd_vs_after, lower_vs_after, upper_vs_after = compute_spectrum_with_confidence(combined_segments_vs_after, fs)
            ax.plot(freqs_vs_after, mean_psd_vs_after, label=f'{drug} After', color=color)
            ax.fill_between(freqs_vs_after, lower_vs_after, upper_vs_after, color=color, alpha=0.3)
            A_after_vs, alpha_after_vs, r_squared_after = fit_one_over_f(freqs_vs_after, mean_psd_vs_after)
            fitted_psd = one_over_f(freqs_vs_after, A_after_vs, alpha_after_vs)
            ax.plot(freqs_vs_after, fitted_psd, label=f'Fit 1/f (α={alpha_after_vs:.2f}, R²={r_squared_after:.2f})', linestyle='--', color='white')

        if combined_segments_vs_before:
            freqs_vs_before, mean_psd_vs_before, lower_vs_before, upper_vs_before = compute_spectrum_with_confidence(combined_segments_vs_before, fs)
            ax.plot(freqs_vs_before, mean_psd_vs_before, label=f'{drug} Baseline', color="yellow")
            ax.fill_between(freqs_vs_before, lower_vs_before, upper_vs_before, color="yellow", alpha=0.3)
            A_before_vs, alpha_before_vs, r_squared_before = fit_one_over_f(freqs_vs_before, mean_psd_vs_before)
            fitted_psd_before = one_over_f(freqs_vs_before, A_before_vs, alpha_before_vs)
            ax.plot(freqs_vs_before, fitted_psd_before, label=f'Fit 1/f (α={alpha_before_vs:.2f}, R²={r_squared_before:.2f})', linestyle='--', color='white')

        ax.set_title(f'VS Individual {individual} - Second 3min', color='white')
        ax.set_xlabel('Frequency (Hz)', color='white')
        ax.set_ylabel('Power', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_facecolor('black')
        ax.legend(loc='upper right')
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

# In your main function, call the plotting function with 'Cocaine' as the drug
def main(pickle_file_path_vs, pickle_file_path_ds):
    # Load the DataFrames from the pickle files
    print("Loading DataFrames from pickle files...")
    df_vs = m_a.load_dataframe(pickle_file_path_vs)
    df_ds = m_a.load_dataframe(pickle_file_path_ds)
    
    fs = TARGET_RATE  # Resampled rate is now 1000 Hz

    # Plot individual spectra for 'Cocaine'
    plot_individual_spectra_second_segment(df_vs, df_ds, fs, 'Cocaine')

# Run the main function
if __name__ == "__main__":
    main(pickle_file_path_vs, pickle_file_path_ds)
