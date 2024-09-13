##### 
# Deconvolution of the signal 
# Events identificaion
# EXtraction of time stamps of events

import numpy as np
import metrics_analysis as m_a
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.cluster import DBSCAN
from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters
from oasis.plotting import simpleaxis
from oasis.oasis_methods import oasisAR1, oasisAR2
from scipy.optimize import curve_fit

def bi_exponential(t, A, tau1, tau2):
    return A * (np.exp(-t / tau1) - np.exp(-t / tau2))
def plot_trace( c, s, b, g, lam,y):
    plt.figure(figsize=(20,4))
    plt.subplot(211)
    plt.plot(b+c, lw=2, label='denoised')
    plt.plot(y, label='data', zorder=-12, c='y')
    plt.legend(ncol=3, frameon=False, loc=(.02,.85))
    simpleaxis(plt.gca())
    plt.subplot(212)
    plt.plot(s/np.max(s), lw=2, label='deconvolved', c='g')
    plt.ylim(0,1.3)
    plt.legend(ncol=3, frameon=False, loc=(.02,.85));
    simpleaxis(plt.gca())
    plt.show()
def peeling_algorithm(signal, template, scale_range=(0.1, 1.3), backtrack_time=0.002, sampling_rate=1000):
    """
    Perform spike inference using a peeling algorithm as described.

    Parameters:
        signal (numpy array): The original fluorescence trace.
        template (numpy array): The stereotypical 1AP-evoked calcium transient waveform.
        scale_range (tuple): The range of scaling factors to apply to the template.
        backtrack_time (float): The amount of time to backtrack after detecting and subtracting an event (in seconds).
        sampling_rate (int): The sampling rate of the signal (in Hz).

    Returns:
        events (list of dicts): A list of detected events with their start time and scaled amplitude.
        residual (numpy array): The final residual trace after peeling.
    """
    residual = np.copy(signal)  # Initialize residual as the original signal
    events = []  # List to store detected events
    template_length = len(template)
    backtrack_samples = int(backtrack_time * sampling_rate)

    while True:
        # Find the peak in the residual, which suggests the next event
        peak_index = np.argmax(residual)
        peak_value = residual[peak_index]
        
        # Break if no significant peaks are found
        if peak_value <= 0:
            break
        
        # Test different scaling factors within the given range
        best_scale = None
        for scale in np.linspace(scale_range[0], scale_range[1], 20):
            scaled_template = scale * template
            start_idx = 0
            end_idx = min(len(residual), start_idx + template_length)
            
            if end_idx <= start_idx:
                continue
            
            # Calculate the integral check
            segment_residual = residual[start_idx:end_idx]
            integral_before = np.sum(segment_residual)
            integral_after = np.sum(segment_residual - scaled_template[:end_idx - start_idx])
            negative_integral_threshold = -0.5 * np.sum(scaled_template[:end_idx - start_idx])
            
            # Check if the subtraction would result in an invalid residual
            if integral_after >= negative_integral_threshold:
                best_scale = scale
                break  # Accept this scale and stop testing further scales
        
        # If no valid scale was found, break the loop
        if best_scale is None:
            break
        
        # Subtract the best scaled template from the residual
        scaled_template = best_scale * template
        residual[start_idx:end_idx] -= scaled_template[:end_idx - start_idx]
        
        # Store the detected event
        events.append({
            'time': peak_index / sampling_rate,
            'amplitude': best_scale * np.max(template)
        })
        
        # Backtrack and continue searching for the next event
        peak_index = max(0, peak_index - backtrack_samples)
        residual = residual[peak_index:]  # Continue from the backtracked position

        # If the residual becomes too short to detect more events, stop
        if len(residual) < template_length:
            break

    return events, residual


PICKLE_FILE_PATH_DS = 'df_combined_vs_all_drugs.pkl'
TEMPLATE_FILE_PATH = 'data/Templates/before_Cocaine_VS_mean_traces_dff.csv'
ORIGINAL_RATE = 1017.252625
TARGET_RATE = 1000  # Target sampling frequency after downsampling

def main(): 
    df = pd.read_pickle(PICKLE_FILE_PATH_DS)

    df['file_prefix'] = df['file'].apply(lambda x: x.split('_')[0])
    #combine files by the animla name
    """
    # Extract the prefix from the "file" field

    # Group the signals by "file_prefix" field and concatenate signals
    grouped_signals_before = df.groupby('file_prefix')['base_before'].apply(lambda x: np.concatenate(x.values)).reset_index()
    grouped_signals_after = df.groupby('file_prefix')['base_after'].apply(lambda x: np.concatenate(x.values)).reset_index()
    """
    # Load or define the template signal
    template = m_a.load_template(TEMPLATE_FILE_PATH, "25")

    template_results = []
    scaled_signals = []
    scaled_deconvolved_signals = []
    # Initialize lists to store results
    evaluation_results = []
    all_scaled_signals = []
    all_scaled_deconvolved_signals = []
    for signal_idx, row in df.iterrows():
        signal_ = row['base_before']
        signal=m_a.resample_signal(signal_, ORIGINAL_RATE,TARGET_RATE)
        template = m_a.resample_signal(template, ORIGINAL_RATE,TARGET_RATE)
        signal=signal[0:10000]+1

        # Evaluate deconvolution
        #snr, rmse, correlation = m_a.evaluate_deconvolution(signal, deconvolved_signal)
        #template_results.append((snr, rmse, correlation))
        
        # Create a time vector for plotting
        time_vector = np.arange(len(signal)) / ORIGINAL_RATE

        c, s, b, g, lam = deconvolve(signal, penalty=1)

        plot_trace( c, s, b, g, lam,signal)    
        print("Inferred spikes:", np.where(s > 0)[0])

    evaluation_results.append(template_results)
    all_scaled_signals.append(scaled_signals)
    all_scaled_deconvolved_signals.append(scaled_deconvolved_signals)

    # Create a dataframe to summarize the evaluation results
    evaluation_df = pd.DataFrame({
        'SNR': [result[0] for template_results in evaluation_results for result in template_results],
        'RMSE': [result[1] for template_results in evaluation_results for result in template_results],
        'Correlation': [result[2] for template_results in evaluation_results for result in template_results]
    })

if __name__ == "__main__":
    main()