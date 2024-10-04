import pandas as pd
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from metrics_analysis import extract_metrics, perform_statistical_tests, plot_metrics,get_normalized_traces, extract_additional_metrics, get_valid_normalized_traces_df, plot_metrics_within
from sklearn.manifold import TSNE
import umap
# Read the DataFrame from the pickle file
pickle_filename = 'data/df_opto_combined_from.pkl'
df = pd.read_pickle(pickle_filename)

# Filter the DataFrame based on 'session_id'
#filtered_df_coc= df[df['session_id'].isin([[3]])]
#filtered_df_cont= df[df['session_id'].isin([[1],[2]])]
#filtered_df_coc['duration'] = filtered_df_coc['duration'].apply(lambda x: x[0])
#filtered_df_coc.reset_index(drop=True, inplace=True)
#filtered_df_cont['duration'] = filtered_df_cont['duration'].apply(lambda x: x[0])
#filtered_df_cont.reset_index(drop=True, inplace=True)

filtered_df_saline= df[df['session']=="Saline"]
filtered_df_coc= df[df['session']=="Cocaine"]
filtered_df_fent= df[df['session']=="Fentanyl"]

filtered_df_coc['duration'] = filtered_df_coc['duration'].apply(lambda x: x[0])
filtered_df_coc.reset_index(drop=True, inplace=True)
filtered_df_saline['duration'] = filtered_df_saline['duration'].apply(lambda x: x[0])
filtered_df_saline.reset_index(drop=True, inplace=True)
filtered_df_fent['duration'] = filtered_df_fent['duration'].apply(lambda x: x[0])
filtered_df_fent.reset_index(drop=True, inplace=True)

# Define the durations and corresponding colors for 'region_id'
region_colors = {
    'DS': 'blue',
    '07': 'green'
}

sampling_rate = 1.017252624511719e+03
start_time = 1.5
end_time = 4.5
start_index = int(start_time * sampling_rate)
end_index = int(end_time * sampling_rate)
norm_end_time = 0.5
norm_end_index = int(norm_end_time * sampling_rate)
time_vector = np.linspace(start_time, end_time, end_index - start_index)
# Create a figure with 4 subplots
# Compare VS vs DS for each filtered DataFrame
durations = [1, 5,10, 25, 50, 100, 250, 1000]
#mean_traces_dict = {'Before_Cocaine': {'DS': {}, 'VS': {}}, 'After_Cocaine': {'DS': {}, 'VS': {}}}
mean_traces_dict = {'Saline': {}, 'Cocaine': {}, 'Fentanyl': {} }

# Create a figure with 2 subplots for each filtered DataFrame
fig, axes = plt.subplots(3, len(durations), figsize=(20, 10))

for i, duration in enumerate(durations):
    for j, (filtered_df, label) in enumerate(zip([filtered_df_saline, filtered_df_coc,filtered_df_fent], ['Saline', 'Cocaine',"Fentanyl"])):
        axes[j, i].set_title(f'{label} - {duration} ms')
        for region_id, color in region_colors.items():
            normalized_traces = get_normalized_traces(filtered_df, region_id, duration, start_index, end_index, norm_end_index)
            if not normalized_traces.empty:
                mean_trace = normalized_traces.apply(pd.Series).mean()
                axes[j, i].plot(time_vector, mean_trace, color=color, label=f'{region_id} mean')
                for k, trace_n in enumerate(normalized_traces):
                    axes[j, i].plot(time_vector, trace_n, color=color, alpha=.1)
                
                #mean_traces_dict[label][region_id][duration] = mean_trace
                mean_traces_dict[label][duration] = mean_trace
        if i == 0:
            axes[j, i].set_ylabel('z-score')
            #axes[j, i].legend()
        #axes[j, i].set_ylim([-40, 320])
        axes[j, i].set_xlabel('Time (s)')

# Adjust layout
plt.tight_layout()
plt.show()
file_path="data/"
for label in mean_traces_dict:
    #for region_id in mean_traces_dict[label]:
    mean_traces_df = pd.DataFrame(mean_traces_dict[label])
    filename = f'{label}_VS_mean_traces_dff_new.csv'
    mean_traces_df.to_csv(filename, index=False)
    print(f'Saved {filename}')
# Show the plot
#plt.savefig("results/plots/traces_opto_VS_DS_zscore.pdf")

sampling_rate = 1.017252624511719e+03
durations = [1,5,10, 25, 50, 100, 250, 1000]
regions = ['DS', 'VS']
metrics = ['FWHM', 'Peak Amplitude', 'Rise Rate', 'Decay Rate', 'AUC',      
        'Half-Decay Time',
        'Max Slope',
        'Rise Time',
        'Decay Time',
        'FDHM',
        'Skewness',
        'Kurtosis',
        'Baseline Drift',
        'Noise Level',
        'Peak-to-Baseline Ratio']
#colors = {'DS': 'blue', 'VS': 'green'}

# Extract metrics for each condition
durations = [1,5,10, 25, 50, 100, 250, 1000]
#regions = ['DS', '07']
# Extract metrics for each condition
data = []
metric_data_list = []  # To store metrics for each trace
filtered_indices = []  # To store indices of traces for each duration and region

for duration in durations:
    for region in regions:
        for filtered_df, session_label in zip([filtered_df_saline, filtered_df_coc, filtered_df_cont], ['After_Cocaine', 'Before_Cocaine']):
            df_region = filtered_df[(filtered_df['duration'] == duration) & (filtered_df['region_id'] == region)]
            if not df_region.empty:
                valid_traces = df_region['data'].apply(lambda x: x[start_index:end_index])
                normalized_traces = valid_traces.apply(lambda x: (x - np.mean(x[:norm_end_index])) / np.std(x[:norm_end_index]))
                metrics_dict = extract_metrics(normalized_traces, sampling_rate)
                additional_metrics_dict = extract_additional_metrics(normalized_traces, sampling_rate)
                for i, trace in enumerate(valid_traces):
                    trace_metrics = {metric: metrics_dict[metric][i] for metric in extract_metrics(normalized_traces, sampling_rate).keys()}
                    additional_trace_metrics = {metric: additional_metrics_dict[metric][i] for metric in extract_additional_metrics(normalized_traces, sampling_rate).keys()}
                    trace_metrics.update(additional_trace_metrics)
                    trace_metrics.update({'Duration': duration, 'Region': region, 'Session': session_label})
                    metric_data_list.append(trace_metrics)

metrics_df = pd.DataFrame(metric_data_list)


# Generate p-values for statistical comparisons
p_values = []

for metric in metrics:
    for duration in durations:
        p_value = perform_statistical_tests(metrics_df, metric,
                                            (metrics_df['Duration'] == duration) & (metrics_df['Region'] == 'VS') & (metrics_df['Session'] == 'After_Cocaine'),
                                            (metrics_df['Duration'] == duration) & (metrics_df['Region'] == 'DS') & (metrics_df['Session'] == 'After_Cocaine'))
        p_values.append({'Metric': metric, 'Group 1': f'VS After_Cocaine {duration}', 'Group 2': f'DS After_Cocaine {duration}', 'p-value': p_value})
        
        p_value = perform_statistical_tests(metrics_df, metric,
                                            (metrics_df['Duration'] == duration) & (metrics_df['Region'] == 'VS') & (metrics_df['Session'] == 'Before_Cocaine'),
                                            (metrics_df['Duration'] == duration) & (metrics_df['Region'] == 'DS') & (metrics_df['Session'] == 'Before_Cocaine'))
        p_values.append({'Metric': metric, 'Group 1': f'VS Before_Cocaine {duration}', 'Group 2': f'DS Before_Cocaine {duration}', 'p-value': p_value})
        
        p_value = perform_statistical_tests(metrics_df, metric,
                                            (metrics_df['Duration'] == duration) & (metrics_df['Region'] == 'VS') & (metrics_df['Session'] == 'After_Cocaine'),
                                            (metrics_df['Duration'] == duration) & (metrics_df['Region'] == 'VS') & (metrics_df['Session'] == 'Before_Cocaine'))
        p_values.append({'Metric': metric, 'Group 1': f'VS After_Cocaine {duration}', 'Group 2': f'VS Before_Cocaine {duration}', 'p-value': p_value})
        
        p_value = perform_statistical_tests(metrics_df, metric,
                                            (metrics_df['Duration'] == duration) & (metrics_df['Region'] == 'DS') & (metrics_df['Session'] == 'After_Cocaine'),
                                            (metrics_df['Duration'] == duration) & (metrics_df['Region'] == 'DS') & (metrics_df['Session'] == 'Before_Cocaine'))
        p_values.append({'Metric': metric, 'Group 1': f'DS After_Cocaine {duration}', 'Group 2': f'DS Before_Cocaine {duration}', 'p-value': p_value})

# Convert p_values to DataFrame and save
p_values_df = pd.DataFrame(p_values)

# Plot each metric
for metric in metrics:
    #plot_metrics(metrics_df, metric, durations, regions, p_values, colors)
    plot_metrics_within(metrics_df, metric, durations, regions, p_values, colors)

# Display the metrics DataFrame and p-values DataFrame for verification
print("Metrics DataFrame:")
print(metrics_df.head())

print("\nP-values DataFrame:")
print(p_values_df.head())
# Save the statistical results to a CSV file
#p_values_df = pd.DataFrame(p_values)
#p_values_df.to_csv('statistical_results.csv', index=False)


filtered_metrics_df=metrics_df.dropna()
filtered_metrics_array = np.array(metric_data_list)

# Dimensionality reduction using UMAP
if filtered_metrics_array.shape[0] > 1:

    # Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=3, perplexity=30, n_iter=300)
    tsne_embedding = tsne.fit_transform(filtered_metrics_array)

    plt.figure(figsize=(12, 8))
    for duration in durations:
        plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], label=f'Duration {duration}', alpha=0.7)
    plt.title('t-SNE Dimensionality Reduction')
    plt.legend()
    plt.show()
else:
    print("Not enough data for dimensionality reduction")


