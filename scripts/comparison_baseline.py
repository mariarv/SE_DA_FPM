import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import welch, coherence

# Paths to the pickle files created previously
pickle_file_path_vs = 'df_combined_vs.pkl'
pickle_file_path_ds = 'df_combined_ds.pkl'

# Function to load the DataFrame from a pickle file
def load_dataframe(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        df = pickle.load(f)
    return df

# Function to compute the power spectrum using Welch's method in the frequency range 0-50 Hz
def compute_power_spectrum(data, fs):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    freqs, power = welch(data, fs, nperseg=1024)
    mask = freqs <= 50  # Focus on frequencies <= 50 Hz
    return freqs[mask], power[mask]

# Function to compute and plot the power spectra for each trace
def plot_power_spectra(df, condition, trace_type):
    plt.figure(figsize=(12, 6))
    for index, row in df.head(10).iterrows():  # Plot only the first 10 trials
        if len(row[trace_type]) > 0:
            freqs, power = compute_power_spectrum(np.array(row[trace_type]), fs=1000)  # Assuming a sampling rate of 1000 Hz
            plt.plot(freqs, power, alpha=0.5, label=f"{row['file']} - {condition}")
    
    plt.title(f"Power Spectrum of {condition} {trace_type.replace('_', ' ')} traces (0-50 Hz)")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.show()

# Function to compute the average power spectrum
def compute_average_power_spectrum(df, trace_type, fs=1000):
    all_power = []
    for index, row in df.iterrows():
        if len(row[trace_type]) > 0:
            _, power = compute_power_spectrum(np.array(row[trace_type]), fs=fs)
            all_power.append(power)
    if all_power:
        average_power = np.mean(all_power, axis=0)
        freqs, _ = compute_power_spectrum(np.array(df.iloc[0][trace_type]), fs=fs)
        return freqs, average_power
    else:
        return None, None

# Function to plot the average power spectrum
def plot_average_power_spectrum(freqs_vs, power_vs, freqs_ds, power_ds, trace_type):
    plt.figure(figsize=(12, 6))
    if freqs_vs is not None and power_vs is not None:
        plt.plot(freqs_vs, power_vs, label=f'VS {trace_type.replace("_", " ")}', color='blue')
    if freqs_ds is not None and power_ds is not None:
        plt.plot(freqs_ds, power_ds, label=f'DS {trace_type.replace("_", " ")}', color='red')
    
    plt.title(f"Average Power Spectrum of {trace_type.replace('_', ' ')} traces (0-50 Hz)")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.show()

# Function to plot auto-correlation
def plot_autocorrelation(df, trace_type, title):
    plt.figure(figsize=(12, 6))
    for index, row in df.head(10).iterrows():  # Plot only the first 10 trials
        if len(row[trace_type]) > 0:
            data = np.array(row[trace_type])
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            plt.plot(autocorr, label=f"{row['file']}")
    plt.title(f"Auto-correlation of {title}")
    plt.xlabel('Lag')
    plt.ylabel('Auto-correlation')
    plt.legend()
    plt.show()

# Function to plot spectrogram
def plot_spectrogram(df, trace_type, fs, title):
    plt.figure(figsize=(12, 6))
    for index, row in df.head(5).iterrows():  # Plot only the first 5 trials to avoid clutter
        if len(row[trace_type]) > 0:
            data = np.array(row[trace_type])
            plt.specgram(data, Fs=fs, NFFT=256, noverlap=128, cmap='jet')
            plt.title(f"Spectrogram of {row['file']} - {title}")
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.show()

# Function to plot coherence
def plot_coherence(df_vs, df_ds, trace_type, fs, title):
    plt.figure(figsize=(12, 6))
    for index, (row_vs, row_ds) in enumerate(zip(df_vs.head(10).iterrows(), df_ds.head(10).iterrows())):  # Plot only the first 10 trials
        if len(row_vs[1][trace_type]) > 0 and len(row_ds[1][trace_type]) > 0:
            data_vs = np.array(row_vs[1][trace_type])
            data_ds = np.array(row_ds[1][trace_type])
            freqs, coh = coherence(data_vs, data_ds, fs, nperseg=1024)
            plt.plot(freqs, coh, label=f"VS-{row_vs[1]['file']} vs DS-{row_ds[1]['file']}")
    plt.title(f"Coherence of {title}")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
    plt.legend()
    plt.show()

# Function to compute Approximate Entropy

# Function to compute Sample Entropy
def sample_entropy(U, m, r):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m)] for i in range(len(U) - m)]
        C = [(len([1 for j in range(len(x)) if _maxdist(x_i, x[j]) <= r]) / (len(x))) for x_i in x]
        return (len(x)) * (len(x) - 1) ** -1 * sum(np.log(C))

    return abs(_phi(m + 1) - _phi(m))

# Function to compute and plot entropy measures
def plot_entropy(df, trace_type, title, m=2, r=0.2):
    samp_entropies = []

    for index, row in df.iterrows():
        if len(row[trace_type]) > 0:
            data = np.array(row[trace_type])
            samp_ent = sample_entropy(data, m, r * np.std(data))
            samp_entropies.append(samp_ent)

    plt.figure(figsize=(12, 6))
    plt.boxplot(samp_entropies)
    plt.title(f"Sample Entropy of {title}")
    plt.show()



# Main function to load data, compute power spectra, and plot results
def main(pickle_file_path_vs, pickle_file_path_ds):
    # Load the DataFrames from the pickle files
    print("Loading DataFrames from pickle files...")
    df_vs = load_dataframe(pickle_file_path_vs)
    df_ds = load_dataframe(pickle_file_path_ds)
    
    # Plot power spectra for individual traces in "before coc" condition
    #print("Plotting power spectra for individual traces in 'before coc' condition...")
   # plot_power_spectra(df_vs, 'VS', 'base_before_coc')
   # plot_power_spectra(df_ds, 'DS', 'base_before_coc')
    
    # Plot power spectra for individual traces in "after coc" condition
    #print("Plotting power spectra for individual traces in 'after coc' condition...")
    #plot_power_spectra(df_vs, 'VS', 'base_after_coc')
   # plot_power_spectra(df_ds, 'DS', 'base_after_coc')
    
    # Compute and plot the average power spectrum in "before coc" condition
    print("Computing and plotting average power spectrum in 'before coc' condition...")
    freqs_vs_before, power_vs_before = compute_average_power_spectrum(df_vs, 'base_before_coc')
    freqs_ds_before, power_ds_before = compute_average_power_spectrum(df_ds, 'base_before_coc')
    plot_average_power_spectrum(freqs_vs_before, power_vs_before, freqs_ds_before, power_ds_before, 'base_before_coc')
    
    # Compute and plot the average power spectrum in "after coc" condition
    print("Computing and plotting average power spectrum in 'after coc' condition...")
    freqs_vs_after, power_vs_after = compute_average_power_spectrum(df_vs, 'base_after_coc')
    freqs_ds_after, power_ds_after = compute_average_power_spectrum(df_ds, 'base_after_coc')
    plot_average_power_spectrum(freqs_vs_after, power_vs_after, freqs_ds_after, power_ds_after, 'base_after_coc')
    
    # Plot auto-correlation
    #print("Plotting auto-correlation for 'before coc' condition...")
    #plot_autocorrelation(df_vs, 'base_before_coc', 'VS before coc')
    #plot_autocorrelation(df_ds, 'base_before_coc', 'DS before coc')
    
    #print("Plotting auto-correlation for 'after coc' condition...")
    #plot_autocorrelation(df_vs, 'base_after_coc', 'VS after coc')
   # plot_autocorrelation(df_ds, 'base_after_coc', 'DS after coc')
    
    # Plot spectrogram
    print("Plotting spectrogram for 'before coc' condition...")
    plot_spectrogram(df_vs, 'base_before_coc', 1000, 'VS before coc')
    plot_spectrogram(df_ds, 'base_before_coc', 1000, 'DS before coc')
    
    print("Plotting spectrogram for 'after coc' condition...")
    plot_spectrogram(df_vs, 'base_after_coc', 1000, 'VS after coc')
    plot_spectrogram(df_ds, 'base_after_coc', 1000, 'DS after coc')
    
    # Plot coherence
    print("Plotting coherence for 'before coc' condition...")
    plot_coherence(df_vs, df_ds, 'base_before_coc', 1000, 'before coc')
    
    print("Plotting coherence for 'after coc' condition...")
    plot_coherence(df_vs, df_ds, 'base_after_coc', 1000, 'after coc')

    # Compute and plot entropy measures for "before coc" condition
    print("Computing and plotting entropy measures for 'before coc' condition...")
    plot_entropy(df_vs, 'base_before_coc', 'VS before coc')
    plot_entropy(df_ds, 'base_before_coc', 'DS before coc')
    
    # Compute and plot entropy measures for "after coc" condition
    print("Computing and plotting entropy measures for 'after coc' condition...")
    plot_entropy(df_vs, 'base_after_coc', 'VS after coc')
    plot_entropy(df_ds, 'base_after_coc', 'DS after coc')

# Run the main function
if __name__ == "__main__":
    main(pickle_file_path_vs, pickle_file_path_ds)
