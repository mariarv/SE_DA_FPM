#########################################################################
# Compare exracted SE waveforms from the recordings and opto stim 
#########################################################################
import pandas as pd
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import metrics_analysis as m_a
import re  # To safely evaluate lists from text
import random
from scipy.stats import mannwhitneyu

# Function to perform resampling test
def resampling_test(temp_df, se_df, metric, num_permutations=1000):
    observed_mean = np.mean(temp_df)  # Calculate observed mean of temp_df

    # Generate permutation samples by randomly sampling from se_df
    permutation_means = []
    for _ in range(num_permutations):
        # Draw a random sample of the same size as temp_df from se_df
        sample = random.choices(se_df[metric], k=len(temp_df))
        permutation_means.append(np.mean(sample))

    # Calculate the p-value as the proportion of permutation means >= observed_mean
    p_value = np.sum(np.array(permutation_means) >= observed_mean) / num_permutations
    return p_value

ORIGINAL_RATE = 1017.252625

start_time = 2.05
end_time = 3.25
start_index = int(start_time * ORIGINAL_RATE)
end_index = int(end_time * ORIGINAL_RATE)

path_waveforms_extracted="/Users/reva/Documents/Python/SE_DA_FPM/results/detected_waveforms_before_coc.pkl"
pickle_filename = 'data/df_opto_combined_from.pkl'
df = pd.read_pickle(pickle_filename)

filtered_df_saline= df[df['session']=="Saline"]
filtered_df_saline['duration'] = filtered_df_saline['duration'].apply(lambda x: x[0])
filtered_df_saline.reset_index(drop=True, inplace=True)

waveforms_df=pd.read_pickle(path_waveforms_extracted)

index_250ms = int(400 * (ORIGINAL_RATE / 1000))  # Calculate index corresponding to 250 ms


durations = [1,5,10, 25, 50, 100, 250, 1000]

metrics = ['FWHM', 'Peak Amplitude', 'Decay Rate', 'AUC',      
        'Half-Decay Time',
        'Max Slope',
        #'Rise Time',
        #'Decay Time',
        'FDHM',
        'Skewness',
        'Kurtosis']

# Create a subplot for each duration
num_durations = len(durations)
fig, axes = plt.subplots(nrows=1, ncols=num_durations+1, figsize=(num_durations * 5, 5), sharey=True)

metric_data_list = []   
# Loop through each waveform in the filtered DataFrame
for waveform_l in waveforms_df['waveforms']:
    waveform_ = [
    trace for trace in waveform_l
    if np.trapz(trace) >= 0.001 and trace[index_250ms]-trace[0]>0.001

    ]    
    trace_lengths = [len(trace) for trace in waveform_]
    min_length = min(trace_lengths)
    truncated_waveform_ = [trace[:min_length] for trace in waveform_]
    # Extract metrics once for the entire waveform
    metrics_dict = m_a.extract_metrics(truncated_waveform_, ORIGINAL_RATE)
    additional_metrics_dict = m_a.extract_additional_metrics(truncated_waveform_, ORIGINAL_RATE)
    truncated_waveform_ = [trace[:min_length] for trace in truncated_waveform_]    
    mean_trace_se = np.mean(truncated_waveform_,axis=0)
   
    # Loop over individual traces in the waveform
    for i, trace in enumerate(truncated_waveform_):
        # Extract the metrics for the current trace
        trace_metrics = {metric: metrics_dict[metric][i] for metric in metrics_dict.keys()}
        additional_trace_metrics = {metric: additional_metrics_dict[metric][i] for metric in additional_metrics_dict.keys()}
        
        # Merge both sets of metrics and add the duration
        trace_metrics.update(additional_trace_metrics)
        
        # Append the trace metrics to the list
        metric_data_list.append(trace_metrics)
        axes[0].plot(trace, color='gray', alpha=0.5, linewidth=0.8)  


    waveform_array_truncated = np.array(truncated_waveform_)
    axes[0].plot(mean_trace_se, color='green', linewidth=2, label='Mean Trace')  # Plot mean trace in red

    #plt.plot(waveform_array_truncated.T)
    #plt.show()
print(len(trace))
# Convert the list of metrics into a DataFrame
metrics_df = pd.DataFrame(metric_data_list)  
print(metrics_df.head())



metric_data_list_opto = []
for i,duration in enumerate(durations):
    # Filter DataFrame based on duration
    filtered_df_saline_ = filtered_df_saline[filtered_df_saline['duration'] == duration]
    valid_traces = filtered_df_saline_['data'].apply(lambda x: m_a.extract_valid_trace(x, ORIGINAL_RATE))
    # Extract metrics once for the entire waveform

    metrics_dict = m_a.extract_metrics(valid_traces, ORIGINAL_RATE)
    additional_metrics_dict = m_a.extract_additional_metrics(valid_traces, ORIGINAL_RATE)
    traces = [np.array(trace) for trace in valid_traces]  # Convert list of traces to a 2D numpy array
    trace_lengths = [len(trace) for trace in traces]
    min_length = min(trace_lengths)
    truncated_waveform_ = [trace[:min_length] for trace in traces]    
    mean_trace = np.mean(truncated_waveform_,axis=0)

    # Loop over individual traces in the waveform
    for j, trace in enumerate(valid_traces):
        # Extract the metrics for the current trace
        trace_metrics = {metric: metrics_dict[metric][j] for metric in metrics_dict.keys()}
        additional_trace_metrics = {metric: additional_metrics_dict[metric][j] for metric in additional_metrics_dict.keys()}
        
        # Merge both sets of metrics and add the duration
        trace_metrics.update(additional_trace_metrics)
        trace_metrics.update({'Duration': duration})
        
        # Append the trace metrics to the list
        metric_data_list_opto.append(trace_metrics)
        axes[i+1].plot(trace, color='gray', alpha=0.5, linewidth=0.8)  
        # Plot each trace with low opacity
    axes[i+1].plot(mean_trace, color='red', linewidth=2, label='Mean Trace')  # Plot mean trace in red
    # Set subplot title and labels
    axes[i+1].set_title(f'Duration: {duration}')
    axes[i+1].set_xlabel('Time')
    axes[i+1].set_ylabel('Amplitude')
plt.tight_layout()
plt.legend()
plt.show()
# Convert the list of metrics into a DataFrame
metric_data_list_opto = pd.DataFrame(metric_data_list_opto)  
print(metric_data_list_opto.head())
print(len(trace))

combined_data = []

# For each metric, add both the original data and the "SE" category data
for metric in metrics:
    print(metric)
    # Create a DataFrame with the original data and category "Original" for the given metric
    temp_df = metric_data_list_opto[['Duration', metric]].copy()
    # Create a second DataFrame for "SE" values, using only metrics_df and matching the same structure
    t_= metrics_df[metrics_df[metric] > 0]
    se_df = pd.DataFrame({
        'Duration': "SE",  # Use the same Duration values as a placeholder
        metric: metrics_df[metric],  # Use the same metric values from metrics_df
    })

    # Combine both DataFrames into a single one for the current metric
    combined_df = pd.concat([se_df, temp_df])
    
    # Plot a swarm plot where each column represents a different duration, grouped by Category
    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='Duration', y=metric,data=combined_df, marker="o", size=5)
    unique_durations = temp_df['Duration'].unique()
    for duration in unique_durations:
        # Get data for the current duration and 'SE'
        duration_data = temp_df[temp_df['Duration'] == duration][metric]
        se_data = se_df[metric]

        # Perform Mann-Whitney U Test
        #stat, p_value = mannwhitneyu(duration_data, se_data, alternative='two-sided')
        p_value = resampling_test(duration_data, se_df,metric)

        # Determine the number of stars for significance level
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = 'ns'

        # Calculate position for annotation
        y_max = max(combined_df[metric])
        y_offset = y_max * 0.1  # Offset above the maximum value for positioning
        x_position = list(unique_durations).index(duration)  # X-position based on duration

        # Add significance text above each comparison
        plt.text(x=x_position+1, y=y_max + y_offset, s=significance, ha='center', va='bottom', fontsize=14, color='black')

    # Customize the plot
    plt.title(f'Swarm Plot of {metric} with Original and SE')
    plt.xlabel('Duration')
    plt.ylabel(metric)
    plt.legend(title=metric)
    
    # Show plot for each metric
    plt.show()