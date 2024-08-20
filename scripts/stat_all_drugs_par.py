import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import logging
import os
from scipy.signal import butter, filtfilt, welch, find_peaks
import EntropyHub as EH
from sklearn.metrics import mutual_info_score
import nolds 
from pyentrp import entropy as ent
from scipy.stats import wilcoxon
import seaborn as sns
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.time_series import TimeSeries
from scipy.signal import butter, filtfilt
import concurrent.futures
from scipy.signal import resample

def false_nearest_neighbors(time_series, max_dim, tau=1, rtol=15, atol=2):
    N = len(time_series)
    fnn_percentages = []

    for m in range(1, max_dim + 1):
        false_neighbors = 0
        total_neighbors = 0

        for i in range(N - (m + 1) * tau):
            # Create the embedded vectors
            x_m = np.array([time_series[i + j * tau] for j in range(m)])
            x_m1 = np.array([time_series[i + j * tau] for j in range(m + 1)])

            # Find the nearest neighbor in m dimensions
            distances = []
            for k in range(N - m * tau):
                if k != i:
                    neighbor = np.array([time_series[k + j * tau] for j in range(m)])
                    distances.append(np.linalg.norm(neighbor - x_m))
            nearest_idx = np.argmin(distances)
            nearest_distance = distances[nearest_idx]

            # Check if this neighbor is false in m+1 dimensions
            nearest_neighbor_m1 = np.array([time_series[nearest_idx + j * tau] for j in range(m + 1)])
            distance_m1 = np.linalg.norm(nearest_neighbor_m1 - x_m1)
            if distance_m1 / nearest_distance > rtol or distance_m1 > atol:
                false_neighbors += 1

            total_neighbors += 1

        fnn_percentages.append(false_neighbors / total_neighbors if total_neighbors > 0 else 0)

    return fnn_percentages

def multiscale_entropy(signal, max_scale):
    m = 2
    r = 0.2 * np.std(signal)
    mse = []
    for scale in range(1, max_scale + 1):
        scaled_signal = signal.reshape(-1, scale).mean(axis=1)
        mse.append(nolds.sampen(scaled_signal, emb_dim=m, tolerance=r))
    return mse
def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def kolmogorov_sinai_entropy(data, bins=100):
    hist_2d, x_edges, y_edges = np.histogram2d(data[:, 0], data[:, 1], bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def reconstruct_phase_space(signal, delay=1):
    """Reconstruct the phase space from a time series."""
    return np.column_stack((signal[:-delay], signal[delay:]))

def compute_fft(signal, fs):
    N = len(signal)
    T = 1.0 / fs
    yf = np.fft.fft(signal)
    xf = np.fft.fftfreq(N, T)[:N // 2]
    return xf, 2.0 / N * np.abs(yf[:N // 2])

def segment_signal(signal, segment_length):
    num_segments = len(signal) // segment_length
    return [signal[i*segment_length:(i+1)*segment_length] for i in range(num_segments)]

def calculate_additional_metrics(signal):
    # Entropy measures
    sample_entropy = nolds.sampen(signal)
    perm_entropy = ent.permutation_entropy(signal, order=3, delay=1, normalize=True)

    # Fractal and chaos measures
    hurst_exponent = nolds.hurst_rs(signal)
    correlation_dimension = nolds.corr_dim(signal, 2)
    lyapunov_exponent = nolds.lyap_r(signal, emb_dim=3)


    return {
        'sample_entropy': sample_entropy,
        'perm_entropy': perm_entropy,
        'hurst_exponent': hurst_exponent,
        'correlation_dimension': correlation_dimension,
        'lyapunov_exponent': lyapunov_exponent,
    }

def average_amplitude(signal):
    return np.mean(np.abs(signal))

def resample_signal(signal, original_fs, target_fs, target_length):
    num_samples = int(len(signal) * target_fs / original_fs)
    if num_samples != target_length:
        signal = resample(signal, target_length)
    return signal



from scipy.fft import fft, fftfreq

def principal_frequency(signal, fs):
    N = len(signal)
    T = 1.0 / fs
    yf = fft(signal)
    xf = fftfreq(N, T)[:N // 2]
    principal_freq = xf[np.argmax(2.0 / N * np.abs(yf[:N // 2]))]
    return principal_freq

def compute_and_plot_stats(df):
    # Initialize a list to store the results
    results = []

    # List of metrics to compare
    metrics = ['KS', 'principal_freq', 'avg_amp', 'sample_entropy', 'perm_entropy', 'hurst_exponent', 
               'correlation_dimension', 'lyapunov_exponent', 'RR', 'DET', 'TT']

    for drug in df['drug'].unique():
        drug_data = df[df['drug'] == drug]
        
        for metric in metrics:
            before_values = drug_data[drug_data['time'] == 'before'][metric].dropna().values
            after_first_values = drug_data[drug_data['time'] == 'after_first'][metric].dropna().values
            after_last_values = drug_data[drug_data['time'] == 'after_last'][metric].dropna().values
            
            min_length = min(len(before_values), len(after_first_values), len(after_last_values))
            before_values = before_values[:min_length]
            after_first_values = after_first_values[:min_length]
            after_last_values = after_last_values[:min_length]
            
            try:
                # Check if inputs are valid arrays of real numbers
                if not (np.isreal(before_values).all() and np.isreal(after_first_values).all() and np.isreal(after_last_values).all()):
                    raise ValueError(f"Invalid input: {metric} for {drug} contains non-real numbers.")
                
                # Perform Wilcoxon signed-rank test
                stat_before_after_first, p_value_before_after_first = wilcoxon(before_values, after_first_values)
                stat_before_after_last, p_value_before_after_last = wilcoxon(before_values, after_last_values)
                stat_after_first_after_last, p_value_after_first_after_last = wilcoxon(after_first_values, after_last_values)
            except ValueError as e:
                if 'zero_method' in str(e) or 'must be an array of real numbers' in str(e):
                    stat_before_after_first, p_value_before_after_first = np.nan, np.nan
                    stat_before_after_last, p_value_before_after_last = np.nan, np.nan
                    stat_after_first_after_last, p_value_after_first_after_last = np.nan, np.nan
                    print(f"Wilcoxon test error for {drug} and {metric}: {e}")
                else:
                    raise

            # Store the results
            results.append({
                'drug': drug,
                'metric': metric,
                'stat_before_after_first': stat_before_after_first,
                'p_value_before_after_first': p_value_before_after_first,
                'stat_before_after_last': stat_before_after_last,
                'p_value_before_after_last': p_value_before_after_last,
                'stat_after_first_after_last': stat_after_first_after_last,
                'p_value_after_first_after_last': p_value_after_first_after_last,
                'mean_before': np.mean(before_values),
                'mean_after_first': np.mean(after_first_values),
                'mean_after_last': np.mean(after_last_values)
            })
            
            # Create a DataFrame for plotting
            plot_data = pd.DataFrame({
                'Value': np.concatenate([before_values, after_first_values, after_last_values]),
                'Condition': ['Before'] * len(before_values) + ['After_First'] * len(after_first_values) + ['After_Last'] * len(after_last_values)
            })
            
            # Ensure 'Value' column is numeric
            plot_data['Value'] = pd.to_numeric(plot_data['Value'], errors='coerce')
            
            # Plot
            plt.figure(figsize=(10, 6))
            sns.violinplot(x='Condition', y='Value', data=plot_data, inner=None)
            sns.stripplot(x='Condition', y='Value', data=plot_data, color='k', alpha=0.5)
            plt.title(f'{metric} for {drug}')
            plt.ylabel(metric)
            
            # Add statistical annotation
            annotation_text = (f'p(before vs after_first): {p_value_before_after_first:.3e}\n'
                               f'p(before vs after_last): {p_value_before_after_last:.3e}\n'
                               f'p(after_first vs after_last): {p_value_after_first_after_last:.3e}')
            plt.annotate(annotation_text, xy=(0.5, 1), xytext=(0.5, 1.1),
                         textcoords='axes fraction', ha='center', fontsize=12)
            plt.savefig(f'results/plots/{metric} for {drug}.pdf')
            plt.close()

    results_df = pd.DataFrame(results)
    return results_df

def process_drug(drug, drug_data, normalization_method='z_score'):
    drug_results = []
    original_fs = 1017.252625  # Original sampling frequency
    target_fs = 100  # Target sampling frequency after downsampling
    target_segment_length = 12000  # Target length after resampling

    for _, row in drug_data.iterrows():
        # Extract and downsample the signal segments
        signal_before = resample_signal(row['base_before'][2000:144140], original_fs, target_fs, target_segment_length)
        signal_after = resample_signal(row['base_after'][0:144140], original_fs, target_fs, target_segment_length)
        signal_after_last = resample_signal(row['base_after'][-144140:], original_fs, target_fs, target_segment_length)

        # Normalize the signals
        if normalization_method == 'z_score':
            signal_before = (signal_before - np.mean(signal_before)) / np.std(signal_before)
            signal_after = (signal_after - np.mean(signal_after)) / np.std(signal_after)
            signal_after_last = (signal_after_last - np.mean(signal_after_last)) / np.std(signal_after_last)
        elif normalization_method == 'min_max':
            signal_before = (signal_before - np.min(signal_before)) / (np.max(signal_before) - np.min(signal_before))
            signal_after = (signal_after - np.min(signal_after)) / (np.max(signal_after) - np.min(signal_after))
            signal_after_last = (signal_after_last - np.min(signal_after_last)) / (np.max(signal_after_last) - np.min(signal_after_last))

        # Process each segment
        segments = {
            'before': signal_before,
            'after_first': signal_after,
            'after_last': signal_after_last
        }
        
        for seg_label, segment in segments.items():
            if len(segment) == 0:
                continue  # Skip empty segments

            KS = kolmogorov_sinai_entropy(np.column_stack((segment, segment)))
            principal_freq = principal_frequency(segment, target_fs)
            avg_amp = average_amplitude(segment)
            additional_metrics = calculate_additional_metrics(segment)

            # RQA settings and computation
            time_series = TimeSeries(segment, embedding_dimension=10, time_delay=10)
            settings = Settings(time_series, neighbourhood=FixedRadius(), similarity_measure=EuclideanMetric)
            computation = RQAComputation.create(settings, verbose=False)
            result = computation.run()
            RR = result.recurrence_rate
            DET = result.determinism
            TT = result.trapping_time

            new_row = pd.DataFrame({
                'drug': [row['drug']],
                'time': [seg_label],
                'KS': [KS],
                'principal_freq': [principal_freq],
                'avg_amp': [avg_amp],
                'sample_entropy': [additional_metrics['sample_entropy']],
                'perm_entropy': [additional_metrics['perm_entropy']],
                'hurst_exponent': [additional_metrics['hurst_exponent']],
                'correlation_dimension': [additional_metrics['correlation_dimension']],
                'lyapunov_exponent': [additional_metrics['lyapunov_exponent']],
                'RR': [RR],
                'DET': [DET],
                'TT': [TT]
            })

            drug_results.append(new_row)

    if not drug_results:
        return pd.DataFrame()  # Return an empty DataFrame if no results

    return pd.concat(drug_results, ignore_index=True)

if __name__ == '__main__':
    # Load the dataframe
    PICKLE_FILE_PATH_DS = 'df_combined_ds_all_drugs.pkl'
    df = pd.read_pickle(PICKLE_FILE_PATH_DS)

    # Choose normalization method
    normalization_method = 'min_max'  # or 'min_max'

    # Process the data with the chosen normalization method in parallel
    drugs = df['drug'].unique()
    results = []

    max_workers = 6  # Set the number of workers to limit resource usage

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_drug, drug, df[df['drug'] == drug], normalization_method): drug for drug in drugs}
        
        for future in concurrent.futures.as_completed(futures):
            drug = futures[future]
            try:
                result = future.result()
                if not result.empty:
                    results.append(result)
                else:
                    print(f"No results for {drug}")
            except Exception as e:
                print(f"Error processing {drug}: {e}")

    if results:
        # Combine results from all drugs
        df_combined = pd.concat(results, ignore_index=True)

        # Perform statistical tests and plot results
        results_df = compute_and_plot_stats(df_combined)

        # Save the results if needed
        results_df.to_csv(f'statistical_comparisons_results_{normalization_method}.csv', index=False)
    else:
        print("No results to process.")