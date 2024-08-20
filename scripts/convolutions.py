import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, find_peaks
from oasis.functions import deconvolve
import logging
import os
from sklearn.metrics import mean_squared_error

# Load the dataframe
PICKLE_FILE_PATH_DS = 'data/df_combined_ds.pkl'
TEMPLATE_FILE_PATH = 'data/Templates/After_Cocaine_DS_mean_traces_dff.csv'
ORIGINAL_RATE = 1017.252625

def load_template(template_file_path, dur):
    if not os.path.exists(template_file_path):
        logging.error(f"File not found: {template_file_path}")
        raise FileNotFoundError(f"File not found: {template_file_path}")
    template_ = pd.read_csv(template_file_path)
    template = template_[dur].values
    logging.info(f"Loaded template from {template_file_path}")
    return np.array(template)

# Extract the "after_coc" column
df = pd.read_pickle(PICKLE_FILE_PATH_DS)
signals = df['base_after_coc']

# Load or define the template signal
templates = [load_template(TEMPLATE_FILE_PATH, "25"), load_template(TEMPLATE_FILE_PATH, "1000")]
template_names = ['25 ms stim', '1000 ms stim']

def bayesian_deconvolution(signal):
    # Perform Bayesian deconvolution using OASIS
    deconvolved_signal, _ = deconvolve(signal)
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

# Initialize lists to store results
evaluation_results = []
all_scaled_signals = []
all_scaled_deconvolved_signals = []

# Perform Bayesian deconvolution and evaluate
for signal_idx, signal in enumerate(signals):
    original_signal, deconvolved_signal = bayesian_deconvolution(signal)
    
    # Scale signals for visualization
    scaled_original_signal = scale_signal(original_signal)
    scaled_deconvolved_signal = scale_signal(deconvolved_signal)
    
    all_scaled_signals.append(scaled_original_signal)
    all_scaled_deconvolved_signals.append(scaled_deconvolved_signal)
    
    # Evaluate deconvolution
    snr, rmse, correlation = evaluate_deconvolution(original_signal, deconvolved_signal)
    evaluation_results.append((snr, rmse, correlation))
    
    # Create a time vector for plotting
    time_vector = np.arange(len(signal)) / ORIGINAL_RATE
    
    # Plot the results for each individual
    plt.figure(figsize=(12, 6))
    plt.plot(time_vector, scaled_original_signal, label='Normalized Signal')
    plt.plot(time_vector, scaled_deconvolved_signal, label='Deconvolved Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Scaled Signal')
    plt.title(f'Individual {signal_idx+1} with Bayesian Deconvolution')
    plt.legend()
    plt.show()

# Create a dataframe to summarize the evaluation results
evaluation_df = pd.DataFrame({
    'Individual': np.arange(1, len(signals) + 1),
    'SNR': [result[0] for result in evaluation_results],
    'RMSE': [result[1] for result in evaluation_results],
    'Correlation': [result[2] for result in evaluation_results]
})

# Display the evaluation dataframe
print(evaluation_df)

# Plot evaluation metrics
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(evaluation_df['Individual'], evaluation_df['SNR'])
plt.xlabel('Individual')
plt.ylabel('SNR')
plt.title('Signal-to-Noise Ratio (SNR)')

plt.subplot(1, 3, 2)
plt.plot(evaluation_df['Individual'], evaluation_df['RMSE'])
plt.xlabel('Individual')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error (RMSE)')

plt.subplot(1, 3, 3)
plt.plot(evaluation_df['Individual'], evaluation_df['Correlation'])
plt.xlabel('Individual')
plt.ylabel('Correlation')
plt.title('Correlation Coefficient')

plt.tight_layout()
plt.show()

# Plot templates
fig, axs = plt.subplots(2, 1, figsize=(15, 8))

for i, template_signal in enumerate(templates):
    time_vector_template = np.arange(len(template_signal)) / ORIGINAL_RATE
    axs[i].plot(time_vector_template, template_signal)
    axs[i].set_title(template_names[i])
    axs[i].set_xlabel('Time (s)')
    axs[i].set_ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Plot zoomed-in version (first 20 seconds) for one individual
fig, axs = plt.subplots(2, 1, figsize=(15, 8))
individual_idx = 0  # Index of the individual to plot

zoomed_time_vector = time_vector[:int(20 * ORIGINAL_RATE)]
for i, template_signal in enumerate(templates):
    scaled_signal = all_scaled_signals[individual_idx]
    scaled_deconvolved_signal = all_scaled_deconvolved_signals[individual_idx]
    axs[i].plot(zoomed_time_vector, scaled_signal[:int(20 * ORIGINAL_RATE)], label='Normalized Signal')
    axs[i].plot(zoomed_time_vector, scaled_deconvolved_signal[:int(20 * ORIGINAL_RATE)], label='Deconvolved Signal')
    axs[i].set_title(f'Zoomed Individual {individual_idx+1} with {template_names[i]}')
    axs[i].set_xlabel('Time (s)')
    axs[i].set_ylabel('Scaled Signal')
    axs[i].legend()

plt.tight_layout()
plt.show()

# Compare the Bayesian inferred template to the original templates
fig, ax = plt.subplots(1, 1, figsize=(15, 6))

# Assuming we extract the inferred template from the deconvolved signals
inferred_template = np.mean([deconvolved_signal for deconvolved_signal in all_scaled_deconvolved_signals], axis=0)

time_vector_inferred = np.arange(len(inferred_template)) / ORIGINAL_RATE

for i, template_signal in enumerate(templates):
    time_vector_template = np.arange(len(template_signal)) / ORIGINAL_RATE
    ax.plot(time_vector_template, scale_signal(template_signal), label=f'Original {template_names[i]} Template')

ax.plot(time_vector_inferred, scale_signal(inferred_template), label='Inferred Bayesian Template', linestyle='--')
ax.set_title('Comparison of Original and Inferred Templates')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.legend()

plt.tight_layout()
plt.show()
