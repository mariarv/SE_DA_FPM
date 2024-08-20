import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
from scipy.signal import resample

def resample_signal(signal, original_fs, target_fs, target_length):
    num_samples = int(len(signal) * target_fs / original_fs)
    if num_samples != target_length:
        signal = resample(signal, target_length)
    return signal

# Load the dataframe
PICKLE_FILE_PATH_DS = 'df_combined_ds_all_drugs.pkl'
ORIGINAL_RATE = 1017.252625
target_fs = 500  # Target sampling frequency after downsampling
target_segment_length = 60000  # Target length after resampling
# Extract the "after_coc" column
df = pd.read_pickle(PICKLE_FILE_PATH_DS)
df['file_prefix'] = df['file'].apply(lambda x: x.split('_')[0])
grouped_signals_before = df.groupby('file_prefix')['base_before'].apply(lambda x: np.concatenate(x.values)).reset_index()
grouped_signals_after = df.groupby('file_prefix')['base_after'].apply(lambda x: np.concatenate(x.values)).reset_index()

for (signal_idx_before, row_before), (signal_idx_after, row_after) in zip(grouped_signals_before.iterrows(), grouped_signals_after.iterrows()):
    # Extract the signal segments
    signal_before = row_before['base_before']
    signal_after = row_after['base_after']
    signal_before = resample_signal(row_before['base_before'][2000:144140], ORIGINAL_RATE, target_fs, target_segment_length)
    signal_after = resample_signal(row_after['base_after'][0:142140], ORIGINAL_RATE, target_fs, target_segment_length)
    # Perform EMD on the "before" signal
    emd_before = EMD()
    IMFs_before = emd_before.emd(signal_before)

    # Perform EMD on the "after" signal
    emd_after = EMD()
    IMFs_after = emd_after.emd(signal_after)

    # Plot the "before" signal and its IMFs
    n_IMFs_before = IMFs_before.shape[0]
    plt.figure(figsize=(12, 9))

    plt.subplot(n_IMFs_before + 1, 1, 1)
    plt.plot(signal_before, 'r')
    plt.title(f"Original 'Before' Signal {signal_idx_before}")

    for i in range(n_IMFs_before):
        plt.subplot(n_IMFs_before + 1, 1, i + 2)
        plt.plot(IMFs_before[i], 'g')
        plt.title(f"IMF {i + 1}")

    plt.tight_layout()
    plt.show()

    # Plot the "after" signal and its IMFs
    n_IMFs_after = IMFs_after.shape[0]
    plt.figure(figsize=(12, 9))

    plt.subplot(n_IMFs_after + 1, 1, 1)
    plt.plot(signal_after, 'b')
    plt.title(f"Original 'After' Signal {signal_idx_after}")

    for i in range(n_IMFs_after):
        plt.subplot(n_IMFs_after + 1, 1, i + 2)
        plt.plot(IMFs_after[i], 'g')
        plt.title(f"IMF {i + 1}")

    plt.tight_layout()
    plt.show()