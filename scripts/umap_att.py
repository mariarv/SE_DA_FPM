import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from metrics_analysis import extract_metrics, plot_traces_above_threshold
import numpy as np

# Load the DataFrame from the pickle file
pickle_filename = 'data/df_opto_combined.pkl'
df_opto_combined = pd.read_pickle(pickle_filename)

# Filter the DataFrame for session_id 1 and 2
df_filtered = df_opto_combined[df_opto_combined['session_id'].isin([[3], [3]])]

df_filtered['session_id'] = df_filtered['session_id'].apply(lambda x: x[0])
df_filtered['duration'] = df_filtered['duration'].apply(lambda x: x[0])

# Define sampling rate and duration times
sampling_rate = 1.017252624511719e+03
start_time = 1.5
end_time = 5
norm_end_time = 0.5
norm_end_index = int(norm_end_time * sampling_rate)
start_index = int(start_time * sampling_rate)
end_index = int(end_time * sampling_rate)
time_vector = np.linspace(start_time, end_time, end_index - start_index)
durations = [25, 50, 100, 250]
regions = ['DS', 'VS']
metrics = ['FWHM', 'Peak Amplitude', 'Rise Rate', 'Decay Rate', 'AUC']
colors = {'DS': 'blue', 'VS': 'green'}

# Normalize traces and extract metrics for each condition
all_traces = []
all_decay_rates = []
filtered_traces = []
filtered_indices = []
metric_data_list = []  # To store metrics for each trace

for duration in durations:
    for region in regions:
        df_region = df_filtered[(df_filtered['duration'] == duration) & (df_filtered['region_id'] == region)]
        if not df_region.empty:
            normalized_traces = []
            valid_traces = df_region['data'].apply(lambda x: x[start_index:end_index] if x[0] <= 0.9 else None).dropna()
            if not valid_traces.empty:
                for trace in valid_traces:
                    baseline_mean = np.mean(trace[:norm_end_index])
                    baseline_std = np.std(trace[:norm_end_index])
                    normalized_trace = (trace - baseline_mean) 
                    normalized_traces.append(normalized_trace)
                normalized_traces = pd.Series(normalized_traces)
                metrics_dict = extract_metrics(normalized_traces, sampling_rate)
                all_traces.extend(normalized_traces.tolist())
                all_decay_rates.extend(metrics_dict['Decay Rate'])
                for i, trace in enumerate(normalized_traces):
                    if metrics_dict['Decay Rate'][i] <= 10:
                        filtered_traces.append(trace)
                        filtered_indices.append((duration, region))
                    trace_metrics = [metrics_dict[metric][i] for metric in metrics]
                    metric_data_list.append(trace_metrics)

# Plot traces with decay rate > 0.12
plot_traces_above_threshold(all_traces, all_decay_rates, 0.12, sampling_rate, time_vector)

# Convert filtered_traces and metrics to numpy arrays
filtered_traces_array = np.array(filtered_traces)
filtered_metrics_array = np.array(metric_data_list)

# Apply UMAP directly on the traces
if filtered_traces_array.shape[0] > 1:
    umap_reducer = umap.UMAP()
    umap_embedding_traces = umap_reducer.fit_transform(filtered_traces_array)

    plt.figure(figsize=(12, 8))
    for region in regions:
        region_indices = [i for i, (d, r) in enumerate(filtered_indices) if r == region]
        plt.scatter(umap_embedding_traces[region_indices, 0], umap_embedding_traces[region_indices, 1], label=f'Region {region}', alpha=0.7)
    plt.title('UMAP Dimensionality Reduction on Traces by Region')
    plt.legend()

    plt.figure(figsize=(12, 8))
    for duration in durations:
        duration_indices = [i for i, (d, r) in enumerate(filtered_indices) if d == duration]
        plt.scatter(umap_embedding_traces[duration_indices, 0], umap_embedding_traces[duration_indices, 1], label=f'Duration {duration}', alpha=0.7)
    plt.title('UMAP Dimensionality Reduction on Traces by Duration')
    plt.legend()
    plt.show()
else:
    print("Not enough data for dimensionality reduction on traces")

# Apply UMAP on the extracted metrics
if filtered_metrics_array.shape[0] > 1:
    umap_reducer = umap.UMAP()
    umap_embedding_metrics = umap_reducer.fit_transform(filtered_metrics_array)

    plt.figure(figsize=(12, 8))
    for region in regions:
        region_indices = [i for i, (d, r) in enumerate(filtered_indices) if r == region]
        plt.scatter(umap_embedding_metrics[region_indices, 0], umap_embedding_metrics[region_indices, 1], label=f'Region {region}', alpha=0.7)
    plt.title('UMAP Dimensionality Reduction on Metrics by Region')
    plt.legend()

    plt.figure(figsize=(12, 8))
    for duration in durations:
        duration_indices = [i for i, (d, r) in enumerate(filtered_indices) if d == duration]
        plt.scatter(umap_embedding_metrics[duration_indices, 0], umap_embedding_metrics[duration_indices, 1], label=f'Duration {duration}', alpha=0.7)
    plt.title('UMAP Dimensionality Reduction on Metrics by Duration')
    plt.legend()
    plt.show()
else:
    print("Not enough data for dimensionality reduction on metrics")

# Apply UMAP on traces for DS and VS separately, coloring by duration
for region in regions:
    filtered_region_traces = [trace for trace, (d, r) in zip(filtered_traces, filtered_indices) if r == region]
    filtered_region_durations = [d for (d, r) in filtered_indices if r == region]
    filtered_region_traces_array = np.array(filtered_region_traces)

    if filtered_region_traces_array.shape[0] > 1:
        umap_reducer = umap.UMAP()
        umap_embedding_region_traces = umap_reducer.fit_transform(filtered_region_traces_array)

        plt.figure(figsize=(12, 8))
        for duration in durations:
            duration_indices = [i for i, d in enumerate(filtered_region_durations) if d == duration]
            plt.scatter(umap_embedding_region_traces[duration_indices, 0], umap_embedding_region_traces[duration_indices, 1], label=f'Duration {duration}', alpha=0.7)
        plt.title(f'UMAP Dimensionality Reduction on {region} Traces by Duration')
        plt.legend()
        plt.show()
    else:
        print(f"Not enough data for dimensionality reduction on {region} traces")

# Apply UMAP on metrics for DS and VS separately, coloring by duration
for region in regions:
    filtered_region_metrics = [metrics for metrics, (d, r) in zip(metric_data_list, filtered_indices) if r == region]
    filtered_region_durations = [d for (d, r) in filtered_indices if r == region]
    filtered_region_metrics_array = np.array(filtered_region_metrics)

    if filtered_region_metrics_array.shape[0] > 1:
        umap_reducer = umap.UMAP()
        umap_embedding_region_metrics = umap_reducer.fit_transform(filtered_region_metrics_array)

        plt.figure(figsize=(12, 8))
        for duration in durations:
            duration_indices = [i for i, d in enumerate(filtered_region_durations) if d == duration]
            plt.scatter(umap_embedding_region_metrics[duration_indices, 0], umap_embedding_region_metrics[duration_indices, 1], label=f'Duration {duration}', alpha=0.7)
        plt.title(f'UMAP Dimensionality Reduction on {region} Metrics by Duration')
        plt.legend()
        plt.show()
    else:
        print(f"Not enough data for dimensionality reduction on {region} metrics")
