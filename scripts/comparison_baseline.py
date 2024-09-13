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
        processed_data = m_a.high_pass_filter(processed_data, fs)
        processed_data = m_a.robust_zscore(processed_data)
    elif trace_type == 'opto_drug':
        processed_data = segment_data(resampled_data, fs,trace_type)['first_3min']  # First 3 minutes
        processed_data = m_a.high_pass_filter(processed_data, fs)
        processed_data = m_a.robust_zscore(processed_data)
    elif trace_type == 'base_after':
        processed_data = segment_data(resampled_data, fs,trace_type)
        for key in processed_data:
            processed_data[key] = processed_data[key]
            processed_data[key] = m_a.high_pass_filter(processed_data[key], fs)
            #processed_data[key] = m_a.remove_trend_polyfit(processed_data[key])
            processed_data[key] = m_a.robust_zscore(processed_data[key])
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
    plt.figure(figsize=(14, 10))
    
    # Define color palette for consistent coloring across animals
    colors = plt.get_cmap('tab10', max(len(segments_vs), len(segments_ds)))
    
    # VS: Calculate and plot mean spectrum with confidence envelopes
    freqs_vs, mean_psd_vs, lower_vs, upper_vs = compute_spectrum_with_confidence(segments_vs, fs)
    plt.plot(freqs_vs, mean_psd_vs, color='blue', label='VS Mean Spectrum')
    plt.fill_between(freqs_vs, lower_vs, upper_vs, color='blue', alpha=0.2, label='VS Confidence Envelope')
    plt.title(f'VS {segment_name.replace("_", " ").capitalize()} - Spectrum with Confidence ({drug_name})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.legend(loc='upper right')
    
    freqs_ds, mean_psd_ds, lower_ds, upper_ds = compute_spectrum_with_confidence(segments_ds, fs)
    plt.plot(freqs_ds, mean_psd_ds, color='orange', label='DS Mean Spectrum')
    plt.fill_between(freqs_ds, lower_ds, upper_ds, color='orange', alpha=0.2, label='DS Confidence Envelope')
    plt.title(f'DS {segment_name.replace("_", " ").capitalize()} - Spectrum with Confidence ({drug_name})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.legend(loc='upper right')

    freqs_ds_base, mean_psd_ds, lower_ds, upper_ds = compute_spectrum_with_confidence(segments_ds, fs)

    
    plt.tight_layout()
    plt.savefig(f'results/Combined_{segment_name.replace("_", " ").capitalize()} - Power Spectrum (dB) ({drug_name}).pdf')

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
# Function to analyze and plot the base_before, opto_drug, and base_after segments for each drug
def analyze_all_conditions_by_drug(df_vs, df_ds, fs):
    drugs_vs = df_vs['drug'].unique()
    drugs_ds = df_ds['drug'].unique()
    drugs = set(drugs_vs).intersection(drugs_ds)  # Consider only common drugs in both datasets

    for drug in drugs:
        print(f"Analyzing for drug: {drug}")
        
        df_vs_drug = df_vs[df_vs['drug'] == drug]
        df_ds_drug = df_ds[df_ds['drug'] == drug]
        

        analyze_base_after_segments(df_vs_drug, df_ds_drug, fs, drug)


def combine_and_plot_condition(df_vs, df_ds, fs, condition_name):
    """
    Combine data across all drugs for the given condition (e.g., 'opto_drug' or 'base_before'),
    and plot the combined segments and spectra with confidence envelopes.
    
    Parameters:
    - df_vs: DataFrame containing the VS data.
    - df_ds: DataFrame containing the DS data.
    - fs: Sampling frequency.
    - condition_name: String indicating the condition to process ('opto_drug' or 'base_before').
    """
    
    # Initialize lists to store the combined segments and IDs for VS and DS
    combined_segments_vs = []
    combined_ids_vs = []
    combined_segments_ds = []
    combined_ids_ds = []

    # Process VS data across all drugs
    for index, row in df_vs.iterrows():
        if len(row[condition_name]) > 0:
            processed_data = process_trace(np.array(row[condition_name]), fs, condition_name)
            combined_segments_vs.append(processed_data)
            combined_ids_vs.append(m_a.get_animal_id(row['file']))

    # Process DS data across all drugs
    for index, row in df_ds.iterrows():
        if len(row[condition_name]) > 0:
            processed_data = process_trace(np.array(row[condition_name]), fs, condition_name)
            combined_segments_ds.append(processed_data)
            combined_ids_ds.append(m_a.get_animal_id(row['file']))

    # Plot the combined data with segments
    plot_segment_analysis(combined_segments_vs, combined_ids_vs, 
                          combined_segments_ds, combined_ids_ds, 
                          fs, f'{condition_name}_combined', 'All Drugs')
    combined_segments_vs_df=pd.DataFrame(combined_segments_vs)
    combined_segments_vs_df.to_csv("data/combined_VS_before_spectra.csv")
    # Plot the combined spectra with confidence envelopes
    plot_spectrum_with_confidence_envelopes(combined_segments_vs, None, combined_segments_ds, None, fs, condition_name, "Combined")


def analyze_base_after_segments(df_vs_drug, df_ds_drug, fs, drug):
    # Initialize dictionaries to store segments and IDs for VS and DS
    segments_vs = {'first_3min': [], 'second_3min': [], 'last_3min': []}
    ids_vs = {'first_3min': [], 'second_3min': [], 'last_3min': []}
    segments_ds = {'first_3min': [], 'second_3min': [], 'last_3min': []}
    ids_ds = {'first_3min': [], 'second_3min': [], 'last_3min': []}
    
    # Process VS data
    for index, row in df_vs_drug.iterrows():
        if len(row['base_after']) > 0:
            segments = process_trace(np.array(row['base_after']), fs, 'base_after')
            for key in segments_vs:
                segments_vs[key].append(segments[key])
                ids_vs[key].append( m_a.get_animal_id(row['file']))

    # Process DS data
    for index, row in df_ds_drug.iterrows():
        if len(row['base_after']) > 0:
            segments = process_trace(np.array(row['base_after']), fs, 'base_after')
            for key in segments_ds:
                segments_ds[key].append(segments[key])
                ids_ds[key].append( m_a.get_animal_id(row['file']))

    # Plot analysis for each segment
    for segment_name in segments_vs.keys():
        plot_segment_analysis(segments_vs[segment_name], ids_vs[segment_name],
                              segments_ds[segment_name], ids_ds[segment_name],
                              fs, segment_name, drug)
        plot_spectrum_with_confidence_envelopes(segments_vs[segment_name], ids_vs[segment_name], segments_ds[segment_name], ids_ds[segment_name], fs, 'opto_drug', drug)



def segment_base_before_data(data, fs):
    segment = data[:int(SEGMENT_DURATION * fs)]
    return segment

def combine_and_plot_spectra_with_envelopes_all_segments(df_vs, df_ds, fs, drug1, drug2):
    """
    Combine data across all drugs for 'base_before', 'opto_drug', and 'base_after' segments,
    and plot the spectra with confidence envelopes. Each drug has a unique color for 'base_before',
    while 'opto_drug' and each segment of 'base_after' are combined across all drugs.
    
    Separate plots for VS and DS conditions.
    
    Parameters:
    - df_vs: DataFrame containing the VS data.
    - df_ds: DataFrame containing the DS data.
    - fs: Sampling frequency.
    """
    
    segments = ['first_3min', 'second_3min', 'last_3min']
    
    # Define color palette for drugs
    drugs = df_vs['drug'].unique()
    color_map = plt.get_cmap('tab10')
    drug_colors = {drug: color_map(i) for i, drug in enumerate(drugs)}
    
    for i, segment_name in enumerate(segments):
        # Initialize combined data for all drugs
        combined_segments_vs_opto = []
        combined_segments_ds_opto = []
        combined_segments_vs_after = []
        combined_segments_ds_after = []

        plt.figure(figsize=(14, 10))
        
        # Plot for VS
        for drug in [drug1,drug2]:
            combined_segments_vs_before = []

            # Process VS data for each drug
            df_vs_drug = df_vs[df_vs['drug'] == drug]

            # Combine VS data for 'base_before'
            for index, row in df_vs_drug.iterrows():
                if len(row['base_before']) > 0:
                    processed_data = process_trace(np.array(row['base_before']), fs, 'base_before')
                    
                    combined_segments_vs_before.append(processed_data)

                    # Combine VS data for 'opto_drug' and 'base_after'
            for index, row in df_vs_drug.iterrows():
                if len(row['opto_drug']) > 0:
                    processed_data = process_trace(np.array(row['opto_drug']), fs, 'opto_drug')
                    combined_segments_vs_opto.append(processed_data)

            for index, row in df_vs_drug.iterrows():
                if len(row['base_after']) > 0:
                    processed_data = process_trace(np.array(row['base_after']), fs, 'base_after')
                    combined_segments_vs_after.append(processed_data[segment_name])


        # Calculate and plot confidence envelopes for combined 'base_after'
            freqs_vs_after, mean_psd_vs_after, lower_vs_after, upper_vs_after = compute_spectrum_with_confidence(combined_segments_vs_after, fs)
            plt.plot(freqs_vs_after, mean_psd_vs_after, label=f'VS {segment_name.capitalize()} ({drug})',  color=drug_colors[drug])
            plt.fill_between(freqs_vs_after, lower_vs_after, upper_vs_after, color=drug_colors[drug], alpha=0.4)
                # Calculate and plot confidence envelopes for 'base_before'
        freqs_vs_before, mean_psd_vs_before, lower_vs_before, upper_vs_before = compute_spectrum_with_confidence(combined_segments_vs_before, fs)
        plt.plot(freqs_vs_before, mean_psd_vs_before, label='VS Baseline Combined', color='purple')
        plt.fill_between(freqs_vs_before, lower_vs_before, upper_vs_before, color='purple', alpha=0.2)

        # Calculate and plot confidence envelopes for combined 'opto_drug'
        freqs_vs_opto, mean_psd_vs_opto, lower_vs_opto, upper_vs_opto = compute_spectrum_with_confidence(combined_segments_vs_opto, fs)
        plt.plot(freqs_vs_opto, mean_psd_vs_opto, label='VS Opto Combined', color='orange')
        plt.fill_between(freqs_vs_opto, lower_vs_opto, upper_vs_opto,color='orange', alpha=0.4)

        plt.title(f'VS Spectra with Confidence Envelopes - Combined Conditions ({segment_name.capitalize()})')
        plt.xlabel('log(Frequency (Hz))')
        plt.ylabel('Power')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/VS Spectra with Confidence Envelopes - Combined Conditions ({segment_name.capitalize()}).pdf")
        plt.show()

        plt.figure(figsize=(14, 10))
        
        # Plot for DS
        for drug in [drug1,drug2]:
            combined_segments_ds_before = []

            # Process DS data for each drug
            df_ds_drug = df_ds[df_ds['drug'] == drug]

            # Combine DS data for 'base_before'
            for index, row in df_ds_drug.iterrows():
                if len(row['base_before']) > 0:
                    processed_data = process_trace(np.array(row['base_before']), fs, 'base_before')
                    combined_segments_ds_before.append(processed_data)

            # Combine DS data for 'opto_drug' and 'base_after'
            for index, row in df_ds_drug.iterrows():
                if len(row['opto_drug']) > 0:
                    processed_data = process_trace(np.array(row['opto_drug']), fs, 'opto_drug')
                    combined_segments_ds_opto.append(processed_data)

            for index, row in df_ds_drug.iterrows():
                if len(row['base_after']) > 0:
                    processed_data = process_trace(np.array(row['base_after']), fs, 'base_after')
                    combined_segments_ds_after.append(processed_data[segment_name])

            # Calculate and plot confidence envelopes for 'base_after'
            freqs_ds_after, mean_psd_ds_after, lower_ds_after, upper_ds_after = compute_spectrum_with_confidence(combined_segments_ds_after, fs)
            plt.plot(freqs_ds_after, mean_psd_ds_after, label=f'DS {segment_name.capitalize()} ({drug})', color=drug_colors[drug])
            plt.fill_between(freqs_ds_after, lower_ds_after, upper_ds_after, color=drug_colors[drug], alpha=0.4)
            # Calculate and plot confidence envelopes for 'base_before'
        freqs_ds_before, mean_psd_ds_before, lower_ds_before, upper_ds_before = compute_spectrum_with_confidence(combined_segments_ds_before, fs)
        plt.plot(freqs_ds_before, mean_psd_ds_before, label='DS Baseline Combined', color="purple")
        plt.fill_between(freqs_ds_before, lower_ds_before, upper_ds_before, color="purple", alpha=0.2)

        # Calculate and plot confidence envelopes for combined 'opto_drug'
        freqs_ds_opto, mean_psd_ds_opto, lower_ds_opto, upper_ds_opto = compute_spectrum_with_confidence(combined_segments_ds_opto, fs)
        plt.plot(freqs_ds_opto, mean_psd_ds_opto, label='DS Opto Combined', color='orange')
        plt.fill_between(freqs_ds_opto, lower_ds_opto, upper_ds_opto, color='orange', alpha=0.4)



        plt.title(f'DS Spectra with Confidence Envelopes - Combined Conditions ({segment_name.capitalize()})')
        plt.xlabel('log(Frequency (Hz))')
        plt.ylabel('Power')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/DS Spectra with Confidence Envelopes - Combined Conditions ({segment_name.capitalize()}).pdf")
        plt.show()


def plot_individual_spectra_second_segment(df_vs, df_ds, fs, drug1, drug2):
    """
    Plot spectra for the second 3-minute segment of 'base_after' condition for two drugs,
    with one subplot per individual. Separate plots for VS and DS conditions.
    
    Parameters:
    - df_vs: DataFrame containing the VS data.
    - df_ds: DataFrame containing the DS data.
    - fs: Sampling frequency.
    - drug1: The first drug to compare.
    - drug2: The second drug to compare.
    """
    
    # Extract individual IDs (assumed to be the first 9 characters of 'file')
    individuals_vs = df_vs['file'].str[:9].unique()
    individuals_ds = df_ds['file'].str[:9].unique()

    # Define color map for the two drugs
    colors = {drug1: 'blue', drug2: 'orange'}

    # VS Plot
    num_individuals_vs = len(individuals_vs)
    fig_vs, axes_vs = plt.subplots(nrows=num_individuals_vs, ncols=1, figsize=(10, 4 * num_individuals_vs))

    for i, individual in enumerate(individuals_vs):
        ax = axes_vs[i] if num_individuals_vs > 1 else axes_vs

        df_vs_individual = df_vs[df_vs['file'].str.startswith(individual)]

        for drug in [drug1, drug2]:
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
                

            # Calculate and plot spectra for the second 3-minute segment of 'base_after'
            if combined_segments_vs_after:
                freqs_vs_after, mean_psd_vs_after, lower_vs_after, upper_vs_after = compute_spectrum_with_confidence(combined_segments_vs_after, fs)
                mean_psd_before_normalized = normalize_curve(mean_psd_vs_after)
                ax.plot(freqs_vs_after, mean_psd_before_normalized, label=f'{drug} After', color=colors[drug])
                ax.fill_between(freqs_vs_after, lower_vs_after, upper_vs_after, color=colors[drug], alpha=0.3)
                A_after_vs, n_after_vs = fit_one_over_f(freqs_vs_after, mean_psd_before_normalized)
                ax.plot(freqs_vs_after, normalize_curve(one_over_f(freqs_vs_after, A_after_vs, n_after_vs)), label=f'Fit 1/f (A={A_after_vs:.2f}, n={n_after_vs:.2f})', linestyle='--')


            if combined_segments_vs_before:
                freqs_vs_before, mean_psd_vs_before, lower_vs_before, upper_vs_before = compute_spectrum_with_confidence(combined_segments_vs_before, fs)
                ax.plot(freqs_vs_before, normalize_curve(mean_psd_vs_before), label=f'{drug} Baseline', color="green")
                ax.fill_between(freqs_vs_before, lower_vs_before, upper_vs_before, color="green", alpha=0.3)
                A_before, n_before = fit_one_over_f(freqs_vs_before, mean_psd_vs_before)
                ax.plot(freqs_vs_before, normalize_curve(one_over_f(freqs_vs_before, A_before, 1)), label=f'Fit 1/f (A={A_before:.2f}, n={n_before:.2f})', linestyle='--')
        
        ax.set_title(f'VS Individual {individual} - Second 3min')
        ax.set_xlabel('log(Frequency (Hz))')
        ax.set_ylabel('Power')
        ax.legend(loc='upper right')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # DS Plot
    num_individuals_ds = len(individuals_ds)
    fig_ds, axes_ds = plt.subplots(nrows=num_individuals_ds, ncols=1, figsize=(10, 4 * num_individuals_ds))

    for i, individual in enumerate(individuals_ds):
        ax = axes_ds[i] if num_individuals_ds > 1 else axes_ds

        df_ds_individual = df_ds[df_ds['file'].str.startswith(individual)]

        for drug in [drug1, drug2]:
            df_ds_drug = df_ds_individual[df_ds_individual['drug'] == drug]

            combined_segments_ds_after = []
            combined_segments_vs_before = []

            for index, row in df_ds_drug.iterrows():
                if len(row['base_after']) > 0:
                    processed_data = process_trace(np.array(row['base_after']), fs, 'base_after')
                    combined_segments_ds_after.append(processed_data['second_3min'])
                if len(row['base_before']) > 0:
                    processed_data = process_trace(np.array(row['base_before']), fs, 'base_before')
                    combined_segments_vs_before.append(processed_data)

            # Calculate and plot spectra for the second 3-minute segment of 'base_after'
            if combined_segments_ds_after:
                freqs_ds_after, mean_psd_ds_after, lower_ds_after, upper_ds_after = compute_spectrum_with_confidence(combined_segments_ds_after, fs)
                ax.plot(freqs_ds_after, mean_psd_ds_after, label=f'{drug} After', color=colors[drug])
                ax.fill_between(freqs_ds_after, lower_ds_after, upper_ds_after, color=colors[drug], alpha=0.3)
            if combined_segments_vs_before:
                freqs_vs_before, mean_psd_vs_before, lower_vs_before, upper_vs_before = compute_spectrum_with_confidence(combined_segments_vs_before, fs)
                ax.plot(freqs_vs_before, mean_psd_vs_before, label=f'{drug} Baseline', color="green")
                ax.fill_between(freqs_vs_before, lower_vs_before, upper_vs_before, color="green", alpha=0.3)

        ax.set_title(f'DS Individual {individual} - Second 3min')
        ax.set_xlabel('log (Frequency (Hz) )')
        ax.set_ylabel('Power')
        ax.legend(loc='upper right')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_pooled_spectra_with_baseline(df_vs, df_ds, fs):
    """
    Plot pooled spectra with confidence envelopes for the baseline and one drug, 
    and repeat this for each drug. The spectra for VS and DS will be plotted side by side.
    
    Parameters:
    - df_vs: DataFrame containing the VS data.
    - df_ds: DataFrame containing the DS data.
    - fs: Sampling frequency.
    """
    drugs = df_vs['drug'].unique()  # Assuming 'drug' is the column name for the drugs

    for drug in drugs:
        # Prepare the figure for each drug
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        fig.suptitle(f'Pooled Spectra with Baseline and {drug}', fontsize=16)
        
        # Initialize lists for segments
        segments_vs = []
        segments_ds = []
        baseline_vs = []
        baseline_ds = []

        # Collect segments for VS and baseline
        df_vs_drug = df_vs[df_vs['drug'] == drug]
        for index, row in df_vs_drug.iterrows():
            if len(row['base_after']) > 0:  # Assuming "base_after" is the segment of interest
                processed_data = process_trace(np.array(row['base_after']), fs, 'base_after')
                segments_vs.append(processed_data['second_3min'])
            if len(row['base_before']) > 0:
                processed_data = process_trace(np.array(row['base_before']), fs, 'base_before')
                baseline_vs.append(processed_data)
        
        # Collect segments for DS and baseline
        df_ds_drug = df_ds[df_ds['drug'] == drug]
        for index, row in df_ds_drug.iterrows():
            if len(row['base_after']) > 0:  # Assuming "base_after" is the segment of interest
                processed_data = process_trace(np.array(row['base_after']), fs, 'base_after')
                segments_ds.append(processed_data['second_3min'])
            if len(row['base_before']) > 0:
                processed_data = process_trace(np.array(row['base_before']), fs, 'base_before')
                baseline_ds.append(processed_data)
        
        # Plot VS spectra
        freqs_vs_baseline, mean_psd_vs_baseline, lower_vs_baseline, upper_vs_baseline = compute_spectrum_with_confidence(baseline_vs, fs)
        freqs_vs, mean_psd_vs, lower_vs, upper_vs = compute_spectrum_with_confidence(segments_vs, fs)
        axes[0].plot(freqs_vs_baseline, mean_psd_vs_baseline, label=f'VS Baseline', color='green')
        axes[0].fill_between(freqs_vs_baseline, lower_vs_baseline, upper_vs_baseline, color='green', alpha=0.3)
        axes[0].plot(freqs_vs, mean_psd_vs, label=f'VS {drug}', color='blue')
        axes[0].fill_between(freqs_vs, lower_vs, upper_vs, color='blue', alpha=0.3)
        axes[0].set_title('VS')
        axes[0].set_xlabel('Log(Frequency (Hz))')
        axes[0].set_ylabel('Power')
        axes[0].legend(loc='upper right')
        axes[0].grid(True)

        # Plot DS spectra
        freqs_ds_baseline, mean_psd_ds_baseline, lower_ds_baseline, upper_ds_baseline = compute_spectrum_with_confidence(baseline_ds, fs)
        freqs_ds, mean_psd_ds, lower_ds, upper_ds = compute_spectrum_with_confidence(segments_ds, fs)
        axes[1].plot(freqs_ds_baseline, mean_psd_ds_baseline, label=f'DS Baseline', color='green')
        axes[1].fill_between(freqs_ds_baseline, lower_ds_baseline, upper_ds_baseline, color='green', alpha=0.3)
        axes[1].plot(freqs_ds, mean_psd_ds, label=f'DS {drug}', color='red')
        axes[1].fill_between(freqs_ds, lower_ds, upper_ds, color='red', alpha=0.3)
        axes[1].set_title('DS')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Power (dB)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True)

        plt.savefig(f'results/Base_vs_({drug}).pdf')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def combine_and_reduce_spectra(df_vs, df_ds, fs):
    """
    Combine spectra for VS, DS, and baseline conditions for each individual, 
    perform PCA, and plot the first three components in 3D.
    - VS conditions are shown as circles and DS conditions as stars.
    - Base VS is shown with a single color, and Base DS with another single color.
    - Same color is used for both VS and DS of the same drug (except for baseline).
    
    Parameters:
    - df_vs: DataFrame containing the VS data.
    - df_ds: DataFrame containing the DS data.
    - fs: Sampling frequency.
    """

    all_spectra = []
    labels = []
    markers = []

    # Define conditions to process
    conditions = ['base_before', 'base_after']

    # Define colors for base conditions
    base_vs_color = 'purple'  # Single color for all VS base
    base_ds_color = 'green'   # Single color for all DS base

    # Store drug-specific colors
    drug_color_map = {}
    color_palette = plt.cm.jet(np.linspace(0, 1, len(df_vs['drug'].unique())))

    for condition in conditions:
        for i, drug in enumerate(df_vs['drug'].unique()):
            # Assign color to the drug
            if drug not in drug_color_map:
                drug_color_map[drug] = color_palette[i]

            # Combine VS data for the condition
            df_vs_drug = df_vs[df_vs['drug'] == drug]
            for index, row in df_vs_drug.iterrows():
                if len(row[condition]) > 0:
                    processed_data = process_trace(np.array(row[condition]), fs, condition)
                    if condition == 'base_before':
                        freq,spectrum = m_a.compute_power_spectrum_dB(processed_data, fs)
                        all_spectra.append(spectrum)
                        labels.append(f'VS_base_{row["file"]}')
                        markers.append('o')  # Circle for VS base
                    else:
                        freq,spectrum = m_a.compute_power_spectrum_dB(processed_data['second_3min'], fs)
                        all_spectra.append(spectrum)
                        labels.append(f'VS_{drug}_{row["file"]}')
                        markers.append('o')  # Circle for VS
            
            # Combine DS data for the condition
            df_ds_drug = df_ds[df_ds['drug'] == drug]
            for index, row in df_ds_drug.iterrows():
                if len(row[condition]) > 0:
                    processed_data = process_trace(np.array(row[condition]), fs, condition)
                    if condition == 'base_before':
                        freq,spectrum = m_a.compute_power_spectrum_dB(processed_data, fs)
                        all_spectra.append(spectrum)
                        labels.append(f'DS_base_{row["file"]}')
                        markers.append('*')  # Star for DS base
                    else:
                        freq,spectrum = m_a.compute_power_spectrum_dB(processed_data['second_3min'], fs)
                        all_spectra.append(spectrum)
                        labels.append(f'DS_{drug}_{row["file"]}')
                        markers.append('*')  # Star for DS

    # Convert all_spectra to a numpy array for PCA
    all_spectra = np.array(all_spectra)

    # Perform PCA
    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(all_spectra)

    # Perform UMAP for dimensionality reduction
    umap_model = umap.UMAP(n_components=3)
    umap_result = umap_model.fit_transform(all_spectra)

    # Plot the PCA result in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, label in enumerate(labels):
        if 'VS_base' in label:
            color = base_vs_color
        elif 'DS_base' in label:
            color = base_ds_color
        else:
            # Assign the same color for both VS and DS of the same drug
            drug = label.split('_')[1]
            color = drug_color_map[drug]
        
        ax.scatter(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], 
                   color=color, marker=markers[i], label=label)

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.set_title('PCA of Combined Spectra')

    # Creating a legend that avoids duplicate entries
    handles, unique_labels = [], []
    for label in np.unique(labels):
        if 'VS_base' in label:
            handle = plt.Line2D([0], [0], marker='o', color=base_vs_color, markersize=8, linestyle='None')
            unique_label = 'VS_base'
        elif 'DS_base' in label:
            handle = plt.Line2D([0], [0], marker='*', color=base_ds_color, markersize=8, linestyle='None')
            unique_label = 'DS_base'
        else:
            drug = label.split('_')[1]
            marker = 'o' if 'VS_' in label else '*'
            handle = plt.Line2D([0], [0], marker=marker, color=drug_color_map[drug], markersize=8, linestyle='None')
            unique_label = drug  # Simplified to just the drug name

        if unique_label not in unique_labels:
            unique_labels.append(unique_label)
            handles.append(handle)
    
    ax.legend(handles, unique_labels, loc='best', bbox_to_anchor=(1, 0.5), title="Condition")
    plt.show()

    # Plot the UMAP result in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, label in enumerate(labels):
        if 'VS_base' in label:
            color = base_vs_color
        elif 'DS_base' in label:
            color = base_ds_color
        else:
            drug = label.split('_')[1]
            color = drug_color_map[drug]
        
        ax.scatter(umap_result[i, 0], umap_result[i, 1], umap_result[i, 2], 
                   color=color, marker=markers[i], label=label)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    ax.set_title('UMAP of Combined Spectra')

    # Adding the same legend to UMAP plot
    ax.legend(handles, unique_labels, loc='best', bbox_to_anchor=(1, 0.5), title="Condition")
    plt.show()



# Main function to load data, process segments by drug, and plot results
def main(pickle_file_path_vs, pickle_file_path_ds):
    # Load the DataFrames from the pickle files
    print("Loading DataFrames from pickle files...")
    df_vs = m_a.load_dataframe(pickle_file_path_vs)
    df_ds = m_a.load_dataframe(pickle_file_path_ds)
    
    fs = TARGET_RATE  # Resampled rate is now 1000 Hz
    
    # Analyze all conditions by drug
    print("Analyzing all conditions by drug...")
    #analyze_all_conditions_by_drug(df_vs, df_ds, fs)
    print("Analyzing combined opto_drug condition for all drugs...")
    #combine_and_plot_condition(df_vs, df_ds, fs, 'opto_drug')
    
    print("Analyzing combined base_before condition for all drugs...")
    combine_and_plot_condition(df_vs, df_ds, fs, 'base_before')
    #print("Analyzing combined conditions for all drugs...")

    drug1 = 'Quinpirole'
    drug2 = 'Raclopride'
    #combine_and_plot_spectra_with_envelopes_all_segments(df_vs, df_ds, fs, drug1, drug2)

    #plot_pooled_spectra_with_baseline(df_vs, df_ds,fs)
    #combine_and_reduce_spectra(df_vs, df_ds, fs)

    plot_individual_spectra_second_segment(df_vs, df_ds, fs, drug1, drug2)

# Run the main function
if __name__ == "__main__":
    main(pickle_file_path_vs, pickle_file_path_ds)
