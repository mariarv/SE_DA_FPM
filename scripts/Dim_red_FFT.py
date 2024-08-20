import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from scipy.fft import fft
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.decomposition import PCA

# Define the function to perform Fourier Transform
def fourier_transform(signal):
    return np.abs(fft(signal))

def resample_signal(signal, original_fs, target_fs, target_length):
    num_samples = int(len(signal) * target_fs / original_fs)
    if num_samples != target_length:
        signal = resample(signal, target_length)
    return signal

def split_trace(trace):
    n = len(trace) // 3
    return [trace[:n], trace[n:2*n], trace[2*n:2*n+n]]


# Example DataFrames with synthetic data
PICKLE_FILE_PATH_DS = 'df_combined_vs_all_drugs.pkl'
ORIGINAL_RATE = 1017.252625
target_fs = 500  # Target sampling frequency after downsampling
target_segment_length = 300000
df = pd.read_pickle(PICKLE_FILE_PATH_DS)
grouped_signals_before = df
grouped_signals_after = df
# Resample signals and collect in a list
signals_before = []
signals_after = []

window_size = 20000  # Window size for the sliding window
step_size = 200    # Step size for the sliding window
signals = []
labels = []
conditions = []

def standardize_trace(trace):
    mean = np.mean(trace)
    std = np.std(trace)
    return (trace - mean)

# Iterate over each pair of before and after signals
for idx, ((signal_idx_before, row_before), (signal_idx_after, row_after)) in enumerate(zip(grouped_signals_before.iterrows(), grouped_signals_after.iterrows())):
    # Extract the signals
    drug=df["drug"][signal_idx_before]
    signal_before = row_before['base_before']
    signal_after = row_after['base_after']
    signal_before = resample_signal(row_before['base_before'][1000:344140], ORIGINAL_RATE, target_fs, target_segment_length)
    signal_after = resample_signal(row_after['base_after'][0:342140], ORIGINAL_RATE, target_fs, target_segment_length)
    signal_after=signal_after[5000:50000]
    signal_before=signal_before[5000:50000]
    # Apply Fourier Transform
    transformed_before = fourier_transform(signal_before)
    transformed_after = fourier_transform(signal_after)
    

    
    signals.append(signal_after)
    labels.append(f'{drug} - After')
    conditions.append('After')

# Convert to numpy array and scale
signals = np.array(signals)
signals=signals.T
qt = QuantileTransformer()
signals_transformed = qt.fit_transform(signals)

#signals_scaled = np.array([standardize_trace(trace) for trace in signals_transformed])

# Apply StandardScaler
scaler = MinMaxScaler()
signals_scaled = scaler.fit_transform(signals)
signals_scaled=signals_scaled.T
means = np.mean(signals_scaled, axis=1)

print(f'Means of the adjusted signals (should be close to 0): {means}')
# Apply UMAP on the combined signals with 3 components

reducer = PCA(n_components=20)
#reducer = PCA(n_components=3, random_state=42)

umap_embedding = reducer.fit_transform(signals_scaled)

# Plot the UMAP results
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a color map for the drugs and conditions
unique_labels = list(set(labels))
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

# Plot each point with the corresponding label color
for i, label in enumerate(labels):
    ax.scatter(umap_embedding[i, 0], umap_embedding[i, 1], umap_embedding[i, 2], 
               color=label_color_map[label], label=label if label not in ax.get_legend_handles_labels()[1] else "")

ax.set_title('UMAP Embedding of Signals')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')

# Create a legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(),loc='center left', bbox_to_anchor=(1.05, 0.5))

plt.show()
"""
df['group'] = df['file'].apply(lambda x: x.split('_')[0])

# Iterate over each group
for group_name, group_df in df.groupby('group'):
    signals = []
    labels = []

    for signal_idx, row in group_df.iterrows():
        
        drug=df["drug"][signal_idx]
        signal_before = row['base_before'][2000:444140]
        signal_after = row['base_after'][0:442140]
        signal_before = resample_signal(row['base_before'][2000:444140], ORIGINAL_RATE, target_fs, target_segment_length)
        signal_after = resample_signal(row['base_after'][0:442140], ORIGINAL_RATE, target_fs, target_segment_length)
        signal_after=signal_after[0:300000]
        signal_before=signal_before[0:100000]
        
        transformed_before = fourier_transform(signal_after)
        transformed_after = fourier_transform(signal_after)
        signals.append(transformed_before)
        labels.append('Before')
        signals.append(transformed_after)
        labels.append(f'{drug} - After')
        
        # Split traces into three parts and inflate dataset
        for part in split_trace(signal_before):
            signals.append(part[0:100000])
            labels.append('Before')
        
        for part in split_trace(signal_after):
            signals.append(part[0:100000])
            labels.append(f'{drug} - After')
        
    # Convert to numpy array
    signals = np.array(signals)
    signals= signals.T
    # Apply QuantileTransformer
    qt = QuantileTransformer()
    signals_transformed =qt.fit_transform(signals)

    # Standardize each trace individually

    # Apply StandardScaler
    scaler = RobustScaler()
    signals_scaled = scaler.fit_transform(signals_transformed)
    #signals_scaled = np.array([standardize_trace(trace) for trace in signals_scaled_])
    signals_scaled= signals_scaled.T
    means = np.mean(signals_scaled, axis=1)
    print(f'Means of the adjusted signals (should be close to 0): {means}')
    # Plot part of the scaled data to see how it looks before and after scaling

    # Apply UMAP on the combined signals with 3 components
    reducer =umap.UMAP(n_components=10)
    umap_embedding = reducer.fit_transform(signals_scaled)

    # Plot the UMAP results
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color map for the labels
    unique_labels = list(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Plot each point with the corresponding label color
    for i, label in enumerate(labels):
        ax.scatter(umap_embedding[i, 0], umap_embedding[i, 1], umap_embedding[i, 2], 
                   color=label_color_map[label], label=label if label not in ax.get_legend_handles_labels()[1] else "")

    ax.set_title(f'UMAP Embedding of Signals for Group {group_name}')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    # Create a legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1.05, 0.5))

    plt.show()
"""