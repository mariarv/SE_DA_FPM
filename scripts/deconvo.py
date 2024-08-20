import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, find_peaks, deconvolve
import logging
import os
from sklearn.metrics import mean_squared_error
from scipy.signal import butter, filtfilt, welch, find_peaks
import EntropyHub as EH
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mutual_info_score
import nolds 
from pyentrp import entropy as ent
from scipy.signal import resample
import sklearn.metrics
from scipy.stats import entropy
from scipy.stats import wilcoxon
import seaborn as sns
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.time_series import TimeSeries
import scipy.stats as stats
from scipy.stats import boxcox, shapiro

from scipy.signal import butter, filtfilt
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
def resample_signal(signal, original_fs, target_fs, target_length):
    num_samples = int(len(signal) * target_fs / original_fs)
    if num_samples != target_length:
        signal = resample(signal, target_length)
    return signal
# Function to design a Butterworth low-pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Function to apply the low-pass filter
def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def multiscale_entropy(signal, max_scale):
    m = 2
    r = 0.2 * np.std(signal)
    mse = []
    for scale in range(1, max_scale + 1):
        scaled_signal = signal.reshape(-1, scale).mean(axis=1)
        mse.append(nolds.sampen(scaled_signal, emb_dim=m, tolerance=r))
    return mse


# Load the dataframe
PICKLE_FILE_PATH_DS = 'df_combined_ds_all_drugs.pkl'
TEMPLATE_FILE_PATH = 'data/Templates/before_Cocaine_VS_mean_traces_dff.csv'
ORIGINAL_RATE = 1017.252625

def load_template(template_file_path, dur):
    if not os.path.exists(template_file_path):
        logging.error(f"File not found: {template_file_path}")
        raise FileNotFoundError(f"File not found: {template_file_path}")
    template_ = pd.read_csv(template_file_path)
    template = template_[dur].values
    logging.info(f"Loaded template from {template_file_path}")
    return np.array(template)

def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def bandpass_filter(signal, lowcut, highcut, fs, order=10):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y
def box_counting_dimension(data, box_sizes):
    """Perform box-counting to estimate the fractal dimension of the given data."""
    counts = []
    for size in box_sizes:
        count = 0
        for i in range(0, data.shape[0], size):
            for j in range(0, data.shape[1], size):
                if np.any(data[i:i + size, j:j + size]):
                    count += 1
        counts.append(count)
    return counts

def fit_fractal_dimension(box_sizes, N):
    log_box_sizes = np.log(box_sizes)
    log_N = np.log(N)
    slope, _ = np.polyfit(log_box_sizes, log_N, 1)
    return slope

def kolmogorov_sinai_entropy(data, bins=100):
    hist_2d, x_edges, y_edges = np.histogram2d(data[:, 0], data[:, 1], bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def poincare_map(data, tau=1):
    """Creates a Poincaré map from the given time series data."""
    return data[:-tau], data[tau:]

def reconstruct_phase_space(signal, delay=1):
    """Reconstruct the phase space from a time series."""
    return np.column_stack((signal[:-delay], signal[delay:]))

def compute_fft(signal, fs):
    N = len(signal)
    T = 1.0 / fs
    yf = np.fft.fft(signal)
    xf = np.fft.fftfreq(N, T)[:N // 2]
    return xf, 2.0 / N * np.abs(yf[:N // 2])
def multiscale_entropy(signal, max_scale):
    m = 2
    r = 0.2 * np.std(signal)
    mse = []

    for scale in range(1, max_scale + 1):
        # Trim the signal to be divisible by the scale factor
        trimmed_signal_length = len(signal) - (len(signal) % scale)
        trimmed_signal = signal[:trimmed_signal_length]
        # Reshape and average
        scaled_signal = trimmed_signal.reshape(-1, scale).mean(axis=1)
        mse.append(nolds.sampen(scaled_signal, emb_dim=m, tolerance=r))
    
    return mse

def compute_and_plot_stats(df):
    # Initialize a list to store the results
    results = []

    # List of metrics to compare
    #metrics = ['KS', 'lyapunov', 'RR', 'DET', 'DIV', 'TT']
    metrics = ['KS']

    for drug in df['drug'].unique():
        drug_data = df[df['drug'] == drug]
        
        for metric in metrics:
            before = drug_data[f'{metric}_before']
            after = drug_data[f'{metric}_after']
            
            try:
                # Check if inputs are valid arrays of real numbers
                if not (np.isreal(before).all() and np.isreal(after).all()):
                    raise ValueError(f"Invalid input: {metric} for {drug} contains non-real numbers.")
                
                # Perform Wilcoxon signed-rank test
                stat, p_value = wilcoxon(before, after)
            except ValueError as e:
                if 'zero_method' in str(e) or 'must be an array of real numbers' in str(e):
                    stat, p_value = np.nan, np.nan
                    print(f"Wilcoxon test error for {drug} and {metric}: {e}")
                else:
                    raise

            # Store the results
            results.append({
                'drug': drug,
                'metric': metric,
                'stat': stat,
                'p_value': p_value,
                'mean_before': np.mean(before),
                'mean_after': np.mean(after)
            })
            
            # Plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=[before, after])
            plt.xticks([0, 1], ['Before', 'After'])
            plt.title(f'{metric} for {drug}')
            plt.ylabel(metric)
            
            # Add statistical annotation
            annotation_text = f'p-value: {p_value:.3e}' if not np.isnan(p_value) else 'p-value: N/A'
            plt.annotate(annotation_text, xy=(0.5, 1), xytext=(0.5, 1.1),
                         textcoords='axes fraction', ha='center', fontsize=12)
            plt.savefig(f'results/plots/{metric} for {drug}.pdf')

    results_df = pd.DataFrame(results)
    return results_df

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
templates = [load_template(TEMPLATE_FILE_PATH, "25"), load_template(TEMPLATE_FILE_PATH, "1000")]
template_names = ['25 ms stim', '1000 ms stim']

grouped_signals_before = df
grouped_signals_after = df

def simple_deconvolution(signal, template_signal):
    # Perform convolution-based template matching
    deconvolved_signal = convolve(signal, template_signal[::-1], mode='same')
    return signal, deconvolved_signal

# Function to scale signals for visualization
def scale_signal(signal):
    return signal / np.max(np.abs(signal))

# Function to calculate evaluation metrics
def evaluate_deconvolution(original_signal, deconvolved_signal):
    # Signal-to-Noise Ratio (SNR)
    snr = np.mean(deconvolved_signal) / np.std(deconvolved_signal)
    
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(mean_squared_error(original_signal, deconvolved_signal))
    
    # Correlation Coefficient
    correlation = np.corrcoef(original_signal, deconvolved_signal)[0, 1]
    
    return snr, rmse, correlation
def recurrence_plot(data, threshold=0.005):
    """Creates a recurrence plot from the given time series data."""
    distances = sklearn.metrics.pairwise_distances(data.reshape(-1, 1))
    return distances < threshold

# Initialize lists to store results
evaluation_results = []
all_scaled_signals = []
all_scaled_deconvolved_signals = []

# Perform deconvolution and evaluate for each template
for template_idx, template_signal in enumerate(templates):
    template_results = []
    scaled_signals = []
    scaled_deconvolved_signals = []
    for signal_idx, row in grouped_signals_after.iterrows():
        signal = row['base_after']
        original_signal, deconvolved_signal = simple_deconvolution(signal, template_signal)
        
        # Scale signals for visualization
        scaled_original_signal = scale_signal(original_signal)
        scaled_deconvolved_signal = scale_signal(deconvolved_signal)
        
        scaled_signals.append(scaled_original_signal)
        scaled_deconvolved_signals.append(scaled_deconvolved_signal)
        
        # Evaluate deconvolution
        snr, rmse, correlation = evaluate_deconvolution(original_signal, deconvolved_signal)
        template_results.append((snr, rmse, correlation))
        
        # Create a time vector for plotting
        time_vector = np.arange(len(signal)) / ORIGINAL_RATE
        """
        # Plot the results for each individual with the current template
        plt.figure(figsize=(12, 6))
        plt.plot(time_vector[0:500000], scaled_original_signal[0:500000], label='Normalized Signal')
        #plt.plot(time_vector[0:100000], scaled_deconvolved_signal[0:100000], label='Deconvolved Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Scaled Signal')
        plt.title(f'Group {signal_idx+1} with {template_names[template_idx]}')
        plt.legend()
        #plt.show()
        """
    evaluation_results.append(template_results)
    all_scaled_signals.append(scaled_signals)
    all_scaled_deconvolved_signals.append(scaled_deconvolved_signals)

# Create a dataframe to summarize the evaluation results
evaluation_df = pd.DataFrame({
    'Template': np.repeat(range(1, len(templates) + 1), len(grouped_signals_after)),
    'Group': np.tile(range(1, len(grouped_signals_after) + 1), len(templates)),
    'SNR': [result[0] for template_results in evaluation_results for result in template_results],
    'RMSE': [result[1] for template_results in evaluation_results for result in template_results],
    'Correlation': [result[2] for template_results in evaluation_results for result in template_results]
})

# Display the evaluation dataframe
print(evaluation_df)
"""
# Plot evaluation metrics
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
for template in range(1, len(templates) + 1):
    plt.plot(evaluation_df[evaluation_df['Template'] == template]['Group'], 
             evaluation_df[evaluation_df['Template'] == template]['SNR'], label=f'Template {template}')
plt.xlabel('Group')
plt.ylabel('SNR')
plt.title('Signal-to-Noise Ratio (SNR)')
plt.legend()

plt.subplot(1, 3, 2)
for template in range(1, len(templates) + 1):
    plt.plot(evaluation_df[evaluation_df['Template'] == template]['Group'], 
             evaluation_df[evaluation_df['Template'] == template]['RMSE'], label=f'Template {template}')
plt.xlabel('Group')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error (RMSE)')
plt.legend()

plt.subplot(1, 3, 3)
for template in range(1, len(templates) + 1):
    plt.plot(evaluation_df[evaluation_df['Template'] == template]['Group'], 
             evaluation_df[evaluation_df['Template'] == template]['Correlation'], label=f'Template {template}')
plt.xlabel('Group')
plt.ylabel('Correlation')
plt.title('Correlation Coefficient')
plt.legend()

#plt.tight_layout()

#plt.show()

# Plot templates
fig, axs = plt.subplots(2, 1, figsize=(15, 8))

for i, template_signal in enumerate(templates):
    time_vector_template = np.arange(len(template_signal)) / ORIGINAL_RATE
    axs[i].plot(time_vector_template, template_signal)
    axs[i].set_title(template_names[i])
    axs[i].set_xlabel('Time (s)')
    axs[i].set_ylabel('Amplitude')

#plt.tight_layout()
##plt.show()

# Plot zoomed-in version (first 20 seconds) for one group
fig, axs = plt.subplots(2, 1, figsize=(15, 8))
group_idx = 0  # Index of the group to plot

zoomed_time_vector = np.arange(int(200 * ORIGINAL_RATE)) / ORIGINAL_RATE
for i in range(len(templates)):
    scaled_signal = all_scaled_signals[i][group_idx]
    scaled_deconvolved_signal = all_scaled_deconvolved_signals[i][group_idx]
    axs[i].plot(zoomed_time_vector, scaled_signal[:int(200 * ORIGINAL_RATE)], label='Normalized Signal')
    axs[i].plot(zoomed_time_vector, scaled_deconvolved_signal[:int(200 * ORIGINAL_RATE)], label='Deconvolved Signal')
    axs[i].set_title(f'Zoomed Group {group_idx+1} with {template_names[i]}')
    axs[i].set_xlabel('Time (s)')
    axs[i].set_ylabel('Scaled Signal')
    axs[i].legend()

#plt.tight_layout()
#plt.show()


for i in range(len(templates)):
    scaled_signal = all_scaled_signals[i][group_idx]
    filtered_baseline_signal = low_pass_filter(scaled_signal, 100, ORIGINAL_RATE)

    f, Pxx = welch(filtered_baseline_signal, ORIGINAL_RATE, nperseg=len(scaled_signal))
    plt.figure()
    plt.loglog(f, Pxx)
    plt.xlim([0, 50])  # Focus on frequencies less than 1 Hz
    plt.title('Power Spectral Density of Baseline Signal (Low Frequencies)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power/Frequency [dB/Hz]')
    plt.show()

    # Find peaks in the PSD to identify dominant frequencies in the low-frequency range
    peaks, properties = find_peaks(Pxx, height=np.mean(Pxx))

    dominant_frequencies = f[peaks]
    dominant_frequencies_low = dominant_frequencies[dominant_frequencies < 1]
    print("Dominant Frequencies (Low):", dominant_frequencies_low)

    
    autocorr = np.correlate(filtered_baseline_signal[0:100000], filtered_baseline_signal[0:100000], mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    plt.figure()
    plt.plot(autocorr)
    plt.title('Autocorrelation of Baseline Signal')
    plt.xlabel('Lag')

    plt.ylabel('Autocorrelation')
    plt.show()
    
df_combined = pd.DataFrame(columns=['drug', 'KS_before', 'KS_after', 'lyapunov_before', 'lyapunov_after',
                                    'RR_before', 'RR_after', 'DET_before', 'DET_after', 
                                    'DIV_before', 'DIV_after', 'TT_before', 'TT_after'])
"""
df_combined = pd.DataFrame(columns=['drug', 'KS_before', 'KS_after'])
for (signal_idx_before, row_before), (signal_idx_after, row_after) in zip(grouped_signals_before.iterrows(), grouped_signals_after.iterrows()):
    # Extract the signal segments
    signal_before = row_before['base_before']
    signal_after = row_after['base_after']
    time = np.linspace(0, 100, 5000)
    signal = np.sin(2 * np.pi * 0.2 * time)  # Base signal
    peaks = np.zeros_like(signal)
    peaks[::50] = 1  # Insert peaks every 500 samples

    # Add Gaussian noise
    noise = np.random.uniform(0, 0.8, size=signal.shape)
    #signal_before =  signal + peaks + noise
    #signal_after = np.convolve(signal_before, np.ones(2000)/2000, mode='valid')
    num_points = min(len(signal_before), len(signal_after),50000)
    original_fs = 1017.252625  # Original sampling frequency
    target_fs = 100  # Target sampling frequency after downsampling
    target_segment_length = 12000  # Target length after resampling

    signal_before = resample_signal(row_before['base_before'][2000:144140], original_fs, target_fs, target_segment_length)
    signal_after = resample_signal(row_after['base_after'][0:142140], original_fs, target_fs, target_segment_length)

    #signal_before = signal_before[2000:62000]
    #signal_after = signal_after[0:60000]
        # Ensure all data is positive by shifting if necessary
    shifted_signal = signal_after - np.min(signal_after) + 1
    signal_after, _ = boxcox(shifted_signal)
    
    shifted_signal_ = signal_before - np.min(signal_before) + 1
    signal_before, _ = boxcox(shifted_signal_)
    lowcut = 0.01  # Low cutoff frequency
    highcut = 100  # High cutoff frequency
    # Apply the low-pass filter
    #signal_after = low_pass_filter(signal_after, highcut, ORIGINAL_RATE)
    #signal_before = low_pass_filter(signal_before, highcut, ORIGINAL_RATE)
    #signal_before = (signal_before - np.mean(signal_before)) / np.std(signal_before)
    #signal_after = (signal_after - np.mean(signal_after)) / np.std(signal_after)
    # Compute the time vector and gradient (velocity)
    time_vector_before = np.arange(len(signal_after)) / ORIGINAL_RATE
    plt.figure()
    plt.plot(time_vector_before,signal_before)
    plt.plot(time_vector_before,signal_after)
    plt.show()    
    v_before = np.gradient(signal_before[0:50000], time_vector_before[0:50000])

    time_vector_after = np.arange(len(signal_after)) / ORIGINAL_RATE
    v_after = np.gradient(signal_after[0:50000], time_vector_after[0:50000])
    drug=df["drug"][signal_idx_before]

    plt.hist(signal_before, bins=30, density=True, alpha=0.6, color='g')
    plt.title('Histogram of Signal')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Plot the theoretical normal distribution on the histogram
    mu, std = stats.norm.fit(signal_before)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()

    # Q-Q Plot
    stats.probplot(signal_before, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Signal')
    plt.show()

    """
    # Find peaks to identify cycles
    peaks_before, _ = find_peaks(signal_before)
    peaks_after, _ = find_peaks(signal_after)
    num_cycles = min(len(peaks_before), len(peaks_after)) - 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plotting the phase space for signal_before
    drug=df["drug"][signal_idx_before]
    ax1.plot( signal_before[0:50000], v_before, label='Phase Space Trace Before')
    ax1.set_title(f'Phase Space Plot Before CoC (Group {signal_idx_before}) for {drug}')
    ax1.set_xlabel('Displacement (x)')
    ax1.set_ylabel('Velocity (v)')
    ax1.grid(True)
    ax1.legend()

    # Plotting the phase space for signal_after
    drug_=df["drug"][signal_idx_before]
    ax2.plot(signal_after[0:50000], v_after, label='Phase Space Trace After')
    ax2.set_title(f'Phase Space Plot After (Group {signal_idx_after}) for {drug_} ')
    ax2.set_xlabel('Displacement (x)')
    ax2.set_ylabel('Velocity (v)')
    ax2.grid(True)
    ax2.legend()

    plt.suptitle(f'Phase Space Plots for Group {signal_idx_before} and {signal_idx_after}')
    plt.savefig(f'results/plots/phase_plots/Phase Space Plots for Group Group {signal_idx_after}) for {drug_}.pdf')
    """
    # Select corresponding cycles from both signals
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i, (data, title) in enumerate(zip([signal_before, signal_after], ["Before", "After"])):
        x, y = poincare_map(data,tau=200)
        axs[i].scatter(x, y, s=2)
        axs[i].set_title(f"Poincaré Map ({title})  for {drug_}  ")
        axs[i].set_xlabel("x(t)")
        axs[i].set_ylabel("x(t + tau)")
    plt.show()
    
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i, (data, title) in enumerate(zip([signal_before, signal_after], ["Before", "After"])):
        recurrence = recurrence_plot(data)
        axs[i].imshow(recurrence, cmap="binary", origin="lower")
        axs[i].set_title(f"Recurrence Plot ({title})  for {drug} ")
    plt.savefig(f'results/plots/RR_plots/Group {signal_idx_before} for {drug}_ds_boxcox.pdf')
    
    """
    plt.subplot(1, 2, 1)
    plt.plot(signal_before, np.gradient(signal_before), ',')
    plt.title('Phase Space Plot Before Intervention (Group 3)')
    plt.xlabel('Displacement (x)')
    plt.ylabel('Velocity (v)')
    plt.grid(True)

    # Phase space plot after intervention
    plt.subplot(1, 2, 2)
    plt.plot(signal_after, np.gradient(signal_after), ',')
    plt.title('Phase Space Plot After Intervention (Group 3)')
    plt.xlabel('Displacement (x)')
    plt.ylabel('Velocity (v)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    print("Fractal dim")
    print([D_before, D_after])
    """
    # Downsample the signals for entropy calculation
    downsampled_signal_before = signal_before[:40000:2]
    downsampled_signal_after = signal_after[:40000:2]
    # Calculate Kolmogorov-Sinai entropy
    KS_before = kolmogorov_sinai_entropy(np.column_stack((downsampled_signal_before, downsampled_signal_before)))
    KS_after = kolmogorov_sinai_entropy(np.column_stack((downsampled_signal_after, downsampled_signal_after)))
    print(drug)

    print("KS measures")
    print([KS_before, KS_after])



    # Calculate Sample Entropy for downsampled signals
    #sampen_before = nolds.lyap_e(downsampled_signal_before, emb_dim=10)
    #sampen_after = nolds.lyap_e(downsampled_signal_after, emb_dim=10)

    #print(f"Corr dim Before: {sampen_before}")
    #print(f"Corr dim After: {sampen_after}")

    #perm_entropy_before = ent.permutation_entropy(downsampled_signal_before, order=3, delay=200, normalize=True)
    #perm_entropy_after = ent.permutation_entropy(downsampled_signal_after, order=3, delay=200, normalize=True)
    #print(f"Permutation Entropy Before: {perm_entropy_before}")
    #print(f"Permutation Entropy After: {perm_entropy_after}")



    time_series_before = TimeSeries(signal_before, embedding_dimension=3, time_delay=300)
    time_series_aft = TimeSeries(signal_after, embedding_dimension=3, time_delay=300)
    """
    fnn_percentages = false_nearest_neighbors(signal_before[0:10000], max_dim=10)

    # Plot the fraction of false nearest neighbors
    plt.plot(range(1, 11), fnn_percentages, marker='o')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Fraction of False Nearest Neighbors')
    plt.title('False Nearest Neighbors Method')
    plt.grid(True)
    plt.show()

    """
    settings_before = Settings(
        time_series_before, 
        neighbourhood=FixedRadius(),  # Adjust radius as needed
        similarity_measure=EuclideanMetric,
        theiler_corrector=0
    )

    settings_after = Settings(
        time_series_aft,
        neighbourhood=FixedRadius(),  # Adjust radius as needed
        similarity_measure=EuclideanMetric,
        theiler_corrector=0
    )
    computation = RQAComputation.create(settings_before, verbose=False)
    result = computation.run()

    # Extract RR and DET
    RR = result.recurrence_rate
    DET = result.determinism
    DIV = result.divergence
    TT = result.trapping_time

    print("Recurrence Rate Before (RR):", RR)
    print("Determinism Before (DET):", DET)
    print("Divergence Before (DIV):", DIV)
    print("Trapping time Before (TT):", TT)

    computation_aft = RQAComputation.create(settings_after, verbose=False)
    result_aft = computation_aft.run()

    # Extract RR and DET
    RR_aft = result_aft.recurrence_rate
    DET_aft = result_aft.determinism
    DIV_aft = result_aft.divergence
    TT_aft = result_aft.trapping_time

    print("Recurrence Rate After (RR):", RR_aft)
    print("Determinism After (DET):", DET_aft)
    print("Divergence After (DIV):", DIV_aft)
    print("Trapping time After (TT):", TT_aft)
    """ 
    xf_before, yf_before = compute_fft(signal_before, ORIGINAL_RATE)
    xf_after, yf_after = compute_fft(signal_after, ORIGINAL_RATE)

    # Plot the FFT results
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.loglog(xf_before, yf_before)
    plt.title('FFT Before Intervention')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.loglog(xf_after, yf_after)
    plt.title('FFT After Intervention')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    new_row = pd.DataFrame({
        'drug': [drug],
        'KS_before': KS_before,
        'KS_after': KS_after
        #'lyapunov_before': sampen_before,
        #'lyapunov_after': sampen_after,
       # 'RR_before': RR,
       # 'RR_after': RR_aft,
       # 'DET_before': DET,
       # 'DET_after': DET_aft,
        #'DIV_before': DIV,
        #'DIV_after': DIV_aft,
        #'TT_before': TT,
        #'TT_after': TT_aft
    })
    df_combined = pd.concat([df_combined, new_row], ignore_index=True)

    # Perform statistical tests and plot results
results_df = compute_and_plot_stats(df_combined)

# Save the results if needed
results_df.to_csv('statistical_comparisons_results.csv', index=False)
"""