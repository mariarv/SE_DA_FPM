import EntropyHub as EH
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from scipy.stats import wilcoxon

from sklearn.metrics import mutual_info_score


file_name = '/Users/reva/Documents/Python/SE_DA_FPM/data/Neuron_SpikeTimes_BeforeCue_Concatenated_all.xlsx'
df = pd.read_excel(file_name)
time_window = 10  # seconds

# Assuming that each column represents the spiking times of a neuron
# and the values are timestamps in seconds
neuron_spiking_times = [df.iloc[:, i].dropna().values for i in range(10)]  # First three neurons

file_name_DA = '/Users/reva/Documents/Python/SE_DA_FPM/data/Neuron_SpikeTimes_BeforeCue.xlsx'
df_da = pd.read_excel(file_name_DA)
name_of_Neuron=df_da.columns

diff_columns = list(set(df.columns).difference(name_of_Neuron))
non_DA_names = diff_columns

session_info=pd.read_excel("/Users/reva/Documents/Python/SE_DA_FPM/data/ids_opto_combined_all.xlsx")
session_info.index = [f'Neuron_{i + 1}' for i in range(len(session_info))]

# Add a "Type" column based on whether the neuron index is in the non-DA list
session_info['Type'] = session_info.index.to_series().apply(lambda x: 'non-DA' if x in non_DA_names else 'DA')

grouped_spikes = {}

# Group `session_info` by `mouse_ind.1` and `mouse_ind` columns
for (ses, mouse), group in session_info.groupby(['mouse_ind.1', 'mouse_ind']):
    neuron_indices = group.index  # Get the neuron names corresponding to this group

    # Use neuron names in `neuron_indices` to select columns in `df`
    grouped_spikes[(ses, mouse)] = df[neuron_indices]

import numpy as np
import matplotlib.pyplot as plt
        # Function to compute average cross-correlation for a set of neurons
        # 
def compute_synchrony_corr(neuron_set,limited_spike_data):
            if len(neuron_set) > 1:
                correlations = []
                for i in range(limited_spike_data.shape[1]):
                    for j in range(i + 1, limited_spike_data.shape[1]):
                        # Drop NaN values from both neuron columns
                        neuron_1 = limited_spike_data.iloc[:, i].dropna()
                        neuron_2 = limited_spike_data.iloc[:, j].dropna()
                        
                        # Ensure equal length after dropping NaNs (required for correlation calculation)
                        min_len = min(len(neuron_1), len(neuron_2))
                        neuron_1 = neuron_1.iloc[:min_len]
                        neuron_2 = neuron_2.iloc[:min_len]
                        
                        # Compute cross-correlation
                        if len(neuron_1) > 1 and len(neuron_2) > 1:  # Ensure enough data points
                            corr = np.corrcoef(neuron_1, neuron_2)[0, 1]
                            correlations.append(corr)

                return np.mean(correlations)
            return None


def compute_synchrony(neuron_set, limited_spike_data, bin_size=0.1, time_limit=10):
    if len(neuron_set) > 1:
        mi_scores = []
        
        # Bin the spike data for each neuron to create binary presence/absence vectors
        n_bins = int(time_limit / bin_size)
        binned_spike_data = pd.DataFrame()

        for neuron in neuron_set:
            neuron_spikes = limited_spike_data[neuron].dropna()  # Remove NaNs from spike data
            # Convert spike times to binary presence/absence in bins
            binned, _ = np.histogram(neuron_spikes, bins=n_bins, range=(0, time_limit))
            binned_spike_data[neuron] = (binned > 0).astype(int)

        # Calculate MI between each pair of neurons
        for i in range(binned_spike_data.shape[1]):
            for j in range(i + 1, binned_spike_data.shape[1]):
                neuron_1 = binned_spike_data.iloc[:, i]
                neuron_2 = binned_spike_data.iloc[:, j]
                
                # Compute MI only if there are enough non-zero elements
                if neuron_1.sum() > 1 and neuron_2.sum() > 1:
                    mi = mutual_info_score(neuron_1, neuron_2)
                    mi_scores.append(mi)

        # Return the average MI for the group of neurons
        return np.mean(mi_scores) if mi_scores else None
    return None

def assess_and_plot_synchrony(grouped_spikes, session_info, time_limit=10):
    synchrony_results = {}

    for group, spike_data in grouped_spikes.items():
        # Retrieve neuron types from `session_info`
        neuron_types = session_info['Type']
        
        # Separate neurons into DA and non-DA based on neuron_types
        da_neurons = [neuron for neuron in spike_data.columns if neuron_types[neuron] == 'DA']
        non_da_neurons = [neuron for neuron in spike_data.columns if neuron_types[neuron] == 'non-DA']
        
        # Limit the data to the first `time_limit` seconds for all neurons in the group
        limited_spike_data = spike_data.apply(lambda x: x[x <= time_limit])

        if limited_spike_data.empty:
            print(f"Skipping group {group}: limited_spike_data is empty.")
            continue
        # Initialize dictionary to store synchrony measures for different categories
        synchrony_results[group] = {}
        
        # Calculate synchrony within DA neurons
        if len(da_neurons) > 1:
            synchrony_results[group]['within_DA'] = compute_synchrony(da_neurons,limited_spike_data)
        else:
            synchrony_results[group]['within_DA'] = None

        # Calculate synchrony within non-DA neurons
        if len(non_da_neurons) > 1:
            synchrony_results[group]['within_non_DA'] = compute_synchrony(non_da_neurons,limited_spike_data)
        else:
            synchrony_results[group]['within_non_DA'] = None

        # Calculate cross-group synchrony (DA vs non-DA) if at least one neuron of each type exists
        if da_neurons and non_da_neurons:
            cross_group_correlations = []
            for da_neuron in da_neurons:
                for non_da_neuron in non_da_neurons:

                    neuron_1 = limited_spike_data[da_neuron]
                    neuron_2 = limited_spike_data[non_da_neuron]
                    neuron_1 = neuron_1.dropna()
                    neuron_2 =neuron_2.dropna()
                    
                    # Ensure equal length after dropping NaNs (required for correlation calculation)
                    min_len = min(len(neuron_1), len(neuron_2))
                    neuron_1 = neuron_1.iloc[:min_len]
                    neuron_2 = neuron_2.iloc[:min_len]
                        
                    corr = np.corrcoef(neuron_1, neuron_2)[0, 1]
                    cross_group_correlations.append(corr)
            synchrony_results[group]['cross_DA_non_DA'] = np.mean(cross_group_correlations)
        else:
            synchrony_results[group]['cross_DA_non_DA'] = None

        # Calculate synchrony for all neurons together if more than one neuron exists
        if len(spike_data.columns) > 1:
            synchrony_results[group]['all_neurons'] = compute_synchrony(spike_data.columns,limited_spike_data)
        else:
            synchrony_results[group]['all_neurons'] = None

        # Plotting the results
        plt.figure(figsize=(10, 6))
        
        # Loop through each neuron to plot spike times
        for i, (neuron_id, spiking_times) in enumerate(limited_spike_data.items()):
            # Plot spikes as vertical lines
            spiking_times_within_window = spiking_times[spiking_times <= time_limit]
            plt.vlines(spiking_times_within_window, i + 0.5, i + 1.5)

        # Formatting the plot with titles including synchrony information
        title = f'Raster Plot for Group {group}\n'
        title += f"Within DA Sync: {synchrony_results[group]['within_DA']}\n"
        title += f"Within non-DA Sync: {synchrony_results[group]['within_non_DA']}\n"
        title += f"Cross DA-nonDA Sync: {synchrony_results[group]['cross_DA_non_DA']}\n"
        title += f"All Neurons Sync: {synchrony_results[group]['all_neurons']}"

        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Neuron Index')
        plt.yticks(range(1, len(spike_data.columns) + 1), [f'Neuron {idx}' for idx in spike_data.columns])
        plt.show()
    
    return synchrony_results


# Function to compute Mutual Information (MI) synchrony
def compute_synchrony_mi(neuron_set, limited_spike_data, bin_size=0.1, time_limit=10):
    if len(neuron_set) > 1:
        mi_scores = []
        
        # Bin the spike data for each neuron to create binary presence/absence vectors
        n_bins = int(time_limit / bin_size)
        binned_spike_data = pd.DataFrame()

        for neuron in neuron_set:
            neuron_spikes = limited_spike_data[neuron].dropna()  # Remove NaNs from spike data
            # Convert spike times to binary presence/absence in bins
            binned, _ = np.histogram(neuron_spikes, bins=n_bins, range=(0, time_limit))
            binned_spike_data[neuron] = (binned > 0).astype(int)

        # Calculate MI between each pair of neurons
        for i in range(binned_spike_data.shape[1]):
            for j in range(i + 1, binned_spike_data.shape[1]):
                neuron_1 = binned_spike_data.iloc[:, i]
                neuron_2 = binned_spike_data.iloc[:, j]
                
                # Compute MI only if there are enough non-zero elements
                if neuron_1.sum() > 1 and neuron_2.sum() > 1:
                    mi = mutual_info_score(neuron_1, neuron_2)
                    mi_scores.append(mi)

        # Return the average MI for the group of neurons
        return np.mean(mi_scores) if mi_scores else None
    return None

# Shuffling and comparison function
def shuffle_isi_spikes(spike_times):
    if len(spike_times) < 2:
        return spike_times  # No shuffling if fewer than 2 spikes
    
    # Calculate ISIs
    isis = np.diff(spike_times)
    
    # Shuffle ISIs
    np.random.shuffle(isis)
    
    # Reconstruct spike train with the first spike time as reference
    shuffled_spike_times = np.cumsum(np.insert(isis, 0, spike_times[0]))
    
    # Convert to Series with NaNs if needed to match length with original
    shuffled_series = pd.Series(shuffled_spike_times)
    if len(shuffled_series) < len(spike_times):
        # Pad with NaNs if shorter
        shuffled_series = shuffled_series.reindex(range(len(spike_times)))
    elif len(shuffled_series) > len(spike_times):
        # Trim if longer
        shuffled_series = shuffled_series[:len(spike_times)]
    
    return shuffled_series

def shuffle_isi_spikes(spike_times):
    if len(spike_times) < 2:
        return spike_times
    isis = np.diff(spike_times)
    np.random.shuffle(isis)
    shuffled_spike_times = np.cumsum(np.insert(isis, 0, spike_times[0]))
    return shuffled_spike_times
def compute_proportion(spikes_ref, spikes_comp, dt):
    if len(spikes_ref) == 0:
        return 0
    time_diffs = np.abs(spikes_ref[:, np.newaxis] - spikes_comp)
    close_spikes = np.sum(np.any(time_diffs <= dt, axis=1))
    proportion = close_spikes / len(spikes_ref)
    return proportion


def calculate_sttc(spikes1, spikes2, dt=0.01, T=10):
    # Convert spikes to NumPy arrays and remove NaNs
    spikes1 = np.asarray(spikes1)
    spikes2 = np.asarray(spikes2)
    spikes1 = spikes1[~np.isnan(spikes1)]
    spikes2 = spikes2[~np.isnan(spikes2)]

    # Check if both spike trains have at least one spike
    if len(spikes1) == 0 or len(spikes2) == 0:
        return np.nan  # Synchrony cannot be computed

    # Compute TA and TB
    TA = (len(spikes1) * 2 * dt) / T
    TB = (len(spikes2) * 2 * dt) / T
    TA = min(TA, 1)
    TB = min(TB, 1)

    # Compute PA and PB
    PA = compute_proportion(spikes1, spikes2, dt)
    PB = compute_proportion(spikes2, spikes1, dt)

    # Calculate STTC
    denominator_A = 1 - PA * TA
    denominator_B = 1 - PB * TB

    # Check denominators
    if denominator_A == 0 or denominator_B == 0:
        return np.nan  # Avoid division by zero

    sttc = 0.5 * ((PA - TA) / denominator_A + (PB - TB) / denominator_B)
    return sttc




# Compute STTC for each neuron pair in a group
def compute_sttc(neuron_set, spike_data, time_limit=10, dt=0.01):
    sttc_values = []
    for i in range(len(neuron_set)):
        for j in range(i + 1, len(neuron_set)):
            neuron_1_spikes = spike_data[neuron_set[i]].dropna().values
            neuron_2_spikes = spike_data[neuron_set[j]].dropna().values
            
            if len(neuron_1_spikes) > 1 and len(neuron_2_spikes) > 1:
                sttc = calculate_sttc(neuron_1_spikes, neuron_2_spikes, dt=dt)
                sttc_values.append(sttc)

    # Return the average STTC for the group
    return np.mean(sttc_values) if sttc_values else None

# Updated assessment function using STTC
# Function for shuffled data STTC
def compute_sttc_shuffled_(neuron_set, spike_data, time_limit=10, dt=0.01):
    shuffled_data = {}
    
    # Create shuffled spike trains for each neuron in neuron_set
    for neuron in neuron_set:
        spike_times = spike_data[neuron].dropna().values
        shuffled_spike_times = shuffle_isi_spikes(spike_times)
        
        # Convert to a Series, reindex to match the original length if necessary
        shuffled_series = pd.Series(shuffled_spike_times)
        
        # Pad with NaNs if shorter to match the original length
        if len(shuffled_series) < len(spike_data[neuron]):
            shuffled_series = shuffled_series.reindex(range(len(spike_data[neuron])), fill_value=np.nan)
        elif len(shuffled_series) > len(spike_data[neuron]):
            shuffled_series = shuffled_series.iloc[:len(spike_data[neuron])]
        
        shuffled_data[neuron] = shuffled_series
    
    # Convert dictionary to DataFrame with padding where needed
    shuffled_df = pd.DataFrame(shuffled_data)
    
    # Calculate STTC for the shuffled data
    shuffled_sttc = compute_sttc(neuron_set, shuffled_df, time_limit, dt)
    
    return shuffled_sttc, shuffled_df

def compute_sttc_shuffled(neuron_set, spike_data, time_limit=10, dt=0.01):
    sttc_values = []
    for i in range(len(neuron_set)):
        for j in range(i + 1, len(neuron_set)):
            neuron1 = neuron_set[i]
            neuron2 = neuron_set[j]
            
            # Get spike times for both neurons
            spikes1 = spike_data[neuron1]
            spikes2 = spike_data[neuron2]
            spikes1 = spikes1[~np.isnan(spikes1)]
            spikes2 = spikes2[~np.isnan(spikes2)]
            # Shuffle spike times of neuron1
            shuffled_spikes1 = shuffle_isi_spikes(spikes1)
            shuffled_spikes2 = shuffle_isi_spikes(spikes2)

            # Compute STTC between shuffled neuron1 and original neuron2
            if len(shuffled_spikes1) > 1 and len(spikes2) > 1:
                sttc = calculate_sttc(shuffled_spikes1, shuffled_spikes2, dt=dt)
                sttc_values.append(sttc)
    # Return the average STTC
    return np.nanmean(sttc_values) if sttc_values else np.nan
# Assessment and plotting function with shuffled comparison
def classify_spikes(spike_times, burst_isi_threshold=0.07):
    if len(spike_times) < 2:
        return spike_times, np.array([])  # All spikes are tonic if less than 2 spikes
    
    isis = np.diff(spike_times)
    burst_indices = np.where(isis < burst_isi_threshold)[0] + 1  # +1 to get the second spike in the ISI

    # Initialize arrays for burst and tonic spikes
    burst_spikes = []
    tonic_spikes = []

    i = 0
    while i < len(spike_times):
        if i in burst_indices:
            # Collect consecutive spikes in a burst
            burst_start = i - 1
            while i < len(spike_times) - 1 and (spike_times[i + 1] - spike_times[i]) < burst_isi_threshold:
                i += 1
            burst_end = i
            burst_spikes.extend(spike_times[burst_start:burst_end + 1])
            i += 1
        else:
            tonic_spikes.append(spike_times[i])
            i += 1

    return np.array(burst_spikes), np.array(tonic_spikes)

# Function to compute STTC for burst or tonic spikes
def compute_sttc_between_neurons(neuron_set, spike_data, time_limit=10, dt=0.01):
    sttc_values = []
    for i in range(len(neuron_set)):
        for j in range(i + 1, len(neuron_set)):
            neuron_1_spikes = spike_data[neuron_set[i]]
            neuron_2_spikes = spike_data[neuron_set[j]]
            neuron_1_spikes = neuron_1_spikes[~np.isnan(neuron_1_spikes)]
            neuron_2_spikes = neuron_2_spikes[~np.isnan(neuron_2_spikes)]
            # Proceed regardless of the number of spikes
            sttc = calculate_sttc(neuron_1_spikes, neuron_2_spikes, dt=dt, T=time_limit)
            sttc_values.append(sttc)
    # Compute the mean STTC, ignoring NaN values
    return np.nanmean(sttc_values) if sttc_values else np.nan


# Function to compute STTC for shuffled data
def compute_sttc_shuffled_between_neurons(neuron_set, spike_data, time_limit=10, dt=0.01):
    sttc_values = []
    for i in range(len(neuron_set)):
        for j in range(i + 1, len(neuron_set)):
            neuron1 = neuron_set[i]
            neuron2 = neuron_set[j]
            
            # Get spike times for both neurons
            spikes1 = spike_data[neuron1]
            spikes2 = spike_data[neuron2]
            spikes1 = spikes1[~np.isnan(spikes1)]
            spikes2 = spikes2[~np.isnan(spikes2)]
            # Shuffle spike times of neuron1
            shuffled_spikes1 = shuffle_isi_spikes(spikes1)
            spikes2 = shuffle_isi_spikes(spikes2)

            # Compute STTC between shuffled neuron1 and original neuron2
            if len(shuffled_spikes1) > 1 and len(spikes2) > 1:
                sttc = calculate_sttc(shuffled_spikes1, spikes2, dt=dt)
                sttc_values.append(sttc)
    # Return the average STTC
    return np.nanmean(sttc_values) if sttc_values else np.nan

# Updated assess function with shuffled data
def assess_synchrony_burst_tonic(grouped_spikes, session_info, time_limit=10, dt=0.01):
    synchrony_results = {}
    data_for_plotting = []  # List to collect data for plotting

    for group, spike_data in grouped_spikes.items():
        neuron_types = session_info['Type']
        da_neurons = [neuron for neuron in spike_data.columns if neuron_types[neuron] == 'DA']
        non_da_neurons = [neuron for neuron in spike_data.columns if neuron_types[neuron] == 'non-DA']

        # Limit data to time window and drop NaNs
        limited_spike_data = spike_data.apply(lambda x: x[x <= time_limit].dropna())

        if limited_spike_data.empty:
            continue

        # Classify spikes into burst and tonic for each neuron
        burst_spike_data = {}
        tonic_spike_data = {}
        for neuron in limited_spike_data.columns:
            spike_times = limited_spike_data[neuron].values
            burst_spikes, tonic_spikes = classify_spikes(spike_times)
            burst_spike_data[neuron] = burst_spikes
            tonic_spike_data[neuron] = tonic_spikes

        synchrony_results[group] = {}

        # Convert 'group' to a string to ensure it's hashable and suitable for plotting
        group_str = str(group)

        # Compute synchrony for burst, tonic, and all spikes
        # For DA neurons
        if len(da_neurons) > 1:
            # Original STTCs
            burst_sttc_da = compute_sttc_between_neurons(da_neurons, burst_spike_data, time_limit, dt)
            tonic_sttc_da = compute_sttc_between_neurons(da_neurons, tonic_spike_data, time_limit, dt)
            all_sttc_da = compute_sttc_between_neurons(da_neurons, limited_spike_data, time_limit, dt)
            synchrony_results[group]['DA_burst'] = burst_sttc_da
            synchrony_results[group]['DA_tonic'] = tonic_sttc_da
            synchrony_results[group]['DA_all'] = all_sttc_da

            # Shuffled STTCs
            burst_sttc_da_shuffled = compute_sttc_shuffled_between_neurons(da_neurons, burst_spike_data, time_limit, dt)
            tonic_sttc_da_shuffled = compute_sttc_shuffled_between_neurons(da_neurons, tonic_spike_data, time_limit, dt)
            all_sttc_da_shuffled = compute_sttc_shuffled_between_neurons(da_neurons, limited_spike_data, time_limit, dt)
            synchrony_results[group]['DA_burst_shuffled'] = burst_sttc_da_shuffled
            synchrony_results[group]['DA_tonic_shuffled'] = tonic_sttc_da_shuffled
            synchrony_results[group]['DA_all_shuffled'] = all_sttc_da_shuffled

            # Collect data for plotting
            data_for_plotting.extend([
                {'Group': group_str, 'Category': 'DA_burst', 'STTC': burst_sttc_da, 'Shuffle': 'Original'},
                {'Group': group_str, 'Category': 'DA_burst', 'STTC': burst_sttc_da_shuffled, 'Shuffle': 'Shuffled'},
                {'Group': group_str, 'Category': 'DA_tonic', 'STTC': tonic_sttc_da, 'Shuffle': 'Original'},
                {'Group': group_str, 'Category': 'DA_tonic', 'STTC': tonic_sttc_da_shuffled, 'Shuffle': 'Shuffled'},
                {'Group': group_str, 'Category': 'DA_all', 'STTC': all_sttc_da, 'Shuffle': 'Original'},
                {'Group': group_str, 'Category': 'DA_all', 'STTC': all_sttc_da_shuffled, 'Shuffle': 'Shuffled'},
            ])
        else:
            synchrony_results[group]['DA_burst'] = None
            synchrony_results[group]['DA_tonic'] = None
            synchrony_results[group]['DA_all'] = None

        # For non-DA neurons
        if len(non_da_neurons) > 1:
            # Original STTCs
            burst_sttc_non_da = compute_sttc_between_neurons(non_da_neurons, burst_spike_data, time_limit, dt)
            tonic_sttc_non_da = compute_sttc_between_neurons(non_da_neurons, tonic_spike_data, time_limit, dt)
            all_sttc_non_da = compute_sttc_between_neurons(non_da_neurons, limited_spike_data, time_limit, dt)
            synchrony_results[group]['non_DA_burst'] = burst_sttc_non_da
            synchrony_results[group]['non_DA_tonic'] = tonic_sttc_non_da
            synchrony_results[group]['non_DA_all'] = all_sttc_non_da

            # Shuffled STTCs
            burst_sttc_non_da_shuffled = compute_sttc_shuffled_between_neurons(non_da_neurons, burst_spike_data, time_limit, dt)
            tonic_sttc_non_da_shuffled = compute_sttc_shuffled_between_neurons(non_da_neurons, tonic_spike_data, time_limit, dt)
            all_sttc_non_da_shuffled = compute_sttc_shuffled_between_neurons(non_da_neurons, limited_spike_data, time_limit, dt)
            synchrony_results[group]['non_DA_burst_shuffled'] = burst_sttc_non_da_shuffled
            synchrony_results[group]['non_DA_tonic_shuffled'] = tonic_sttc_non_da_shuffled
            synchrony_results[group]['non_DA_all_shuffled'] = all_sttc_non_da_shuffled

            # Collect data for plotting
            data_for_plotting.extend([
                {'Group': group_str, 'Category': 'non_DA_burst', 'STTC': burst_sttc_non_da, 'Shuffle': 'Original'},
                {'Group': group_str, 'Category': 'non_DA_burst', 'STTC': burst_sttc_non_da_shuffled, 'Shuffle': 'Shuffled'},
                {'Group': group_str, 'Category': 'non_DA_tonic', 'STTC': tonic_sttc_non_da, 'Shuffle': 'Original'},
                {'Group': group_str, 'Category': 'non_DA_tonic', 'STTC': tonic_sttc_non_da_shuffled, 'Shuffle': 'Shuffled'},
                {'Group': group_str, 'Category': 'non_DA_all', 'STTC': all_sttc_non_da, 'Shuffle': 'Original'},
                {'Group': group_str, 'Category': 'non_DA_all', 'STTC': all_sttc_non_da_shuffled, 'Shuffle': 'Shuffled'},
            ])
        else:
            synchrony_results[group]['non_DA_burst'] = None
            synchrony_results[group]['non_DA_tonic'] = None
            synchrony_results[group]['non_DA_all'] = None

        # Cross-group synchrony
        if da_neurons and non_da_neurons:
            # Original STTCs
            # Burst spikes
            burst_sttc_cross = []
            for da_neuron in da_neurons:
                for non_da_neuron in non_da_neurons:
                    neuron_1_spikes = burst_spike_data[da_neuron]
                    neuron_2_spikes = burst_spike_data[non_da_neuron]
                    if len(neuron_1_spikes) > 1 and len(neuron_2_spikes) > 1:
                        sttc = calculate_sttc(neuron_1_spikes, neuron_2_spikes, dt=dt)
                        burst_sttc_cross.append(sttc)
            cross_burst_sttc = np.mean(burst_sttc_cross) if burst_sttc_cross else None
            synchrony_results[group]['cross_burst'] = cross_burst_sttc

            # Tonic spikes
            tonic_sttc_cross = []
            for da_neuron in da_neurons:
                for non_da_neuron in non_da_neurons:
                    neuron_1_spikes = tonic_spike_data[da_neuron]
                    neuron_2_spikes = tonic_spike_data[non_da_neuron]
                    if len(neuron_1_spikes) > 1 and len(neuron_2_spikes) > 1:
                        sttc = calculate_sttc(neuron_1_spikes, neuron_2_spikes, dt=dt)
                        tonic_sttc_cross.append(sttc)
            cross_tonic_sttc = np.mean(tonic_sttc_cross) if tonic_sttc_cross else None
            synchrony_results[group]['cross_tonic'] = cross_tonic_sttc

            # All spikes
            all_sttc_cross = []
            for da_neuron in da_neurons:
                for non_da_neuron in non_da_neurons:
                    neuron_1_spikes = limited_spike_data[da_neuron].values
                    neuron_2_spikes = limited_spike_data[non_da_neuron].values
                    if len(neuron_1_spikes) > 1 and len(neuron_2_spikes) > 1:
                        sttc = calculate_sttc(neuron_1_spikes, neuron_2_spikes, dt=dt)
                        all_sttc_cross.append(sttc)
            cross_all_sttc = np.mean(all_sttc_cross) if all_sttc_cross else None
            synchrony_results[group]['cross_all'] = cross_all_sttc

            # Shuffled STTCs
            # Burst spikes
            burst_sttc_cross_shuffled = []
            for da_neuron in da_neurons:
                for non_da_neuron in non_da_neurons:
                    neuron_1_spikes = shuffle_isi_spikes(burst_spike_data[da_neuron])
                    neuron_2_spikes = shuffle_isi_spikes(burst_spike_data[non_da_neuron])
                    if len(neuron_1_spikes) > 1 and len(neuron_2_spikes) > 1:
                        sttc = calculate_sttc(neuron_1_spikes, neuron_2_spikes, dt=dt)
                        burst_sttc_cross_shuffled.append(sttc)
            cross_burst_sttc_shuffled = np.mean(burst_sttc_cross_shuffled) if burst_sttc_cross_shuffled else None
            synchrony_results[group]['cross_burst_shuffled'] = cross_burst_sttc_shuffled

            # Tonic spikes
            tonic_sttc_cross_shuffled = []
            for da_neuron in da_neurons:
                for non_da_neuron in non_da_neurons:
                    neuron_1_spikes = shuffle_isi_spikes(tonic_spike_data[da_neuron])
                    neuron_2_spikes = shuffle_isi_spikes(tonic_spike_data[non_da_neuron])
                    if len(neuron_1_spikes) > 1 and len(neuron_2_spikes) > 1:
                        sttc = calculate_sttc(neuron_1_spikes, neuron_2_spikes, dt=dt)
                        tonic_sttc_cross_shuffled.append(sttc)
            cross_tonic_sttc_shuffled = np.mean(tonic_sttc_cross_shuffled) if tonic_sttc_cross_shuffled else None
            synchrony_results[group]['cross_tonic_shuffled'] = cross_tonic_sttc_shuffled

            # All spikes
            all_sttc_cross_shuffled = []
            for da_neuron in da_neurons:
                for non_da_neuron in non_da_neurons:
                    neuron_1_spikes = shuffle_isi_spikes(limited_spike_data[da_neuron].values)
                    neuron_2_spikes = shuffle_isi_spikes(limited_spike_data[non_da_neuron].values)
                    if len(neuron_1_spikes) > 1 and len(neuron_2_spikes) > 1:
                        sttc = calculate_sttc(neuron_1_spikes, neuron_2_spikes, dt=dt)
                        all_sttc_cross_shuffled.append(sttc)
            cross_all_sttc_shuffled = np.mean(all_sttc_cross_shuffled) if all_sttc_cross_shuffled else None
            synchrony_results[group]['cross_all_shuffled'] = cross_all_sttc_shuffled

            # Collect data for plotting
            data_for_plotting.extend([
                {'Group': group_str, 'Category': 'cross_burst', 'STTC': cross_burst_sttc, 'Shuffle': 'Original'},
                {'Group': group_str, 'Category': 'cross_burst', 'STTC': cross_burst_sttc_shuffled, 'Shuffle': 'Shuffled'},
                {'Group': group_str, 'Category': 'cross_tonic', 'STTC': cross_tonic_sttc, 'Shuffle': 'Original'},
                {'Group': group_str, 'Category': 'cross_tonic', 'STTC': cross_tonic_sttc_shuffled, 'Shuffle': 'Shuffled'},
                {'Group': group_str, 'Category': 'cross_all', 'STTC': all_sttc_cross, 'Shuffle': 'Original'},
                {'Group': group_str, 'Category': 'cross_all', 'STTC': all_sttc_cross_shuffled, 'Shuffle': 'Shuffled'},
            ])
        else:
            synchrony_results[group]['cross_burst'] = None
            synchrony_results[group]['cross_tonic'] = None
            synchrony_results[group]['cross_all'] = None

        # Plotting code for the spike trains remains the same
        # (Refer to the previous code for plotting the three subplots)

    # After processing all groups, create a DataFrame for plotting
    df_plot = pd.DataFrame(data_for_plotting)

    # Remove entries with None STTC values
    df_plot = df_plot.dropna(subset=['STTC'])

    # Return both synchrony_results and df_plot
    return synchrony_results, df_plot

# Run the updated analysis
synchrony_results, df_plot = assess_synchrony_burst_tonic(grouped_spikes, session_info)

def compute_firing_rates(grouped_spikes, session_info, time_limit=10):
    firing_rates = []
    for group, spike_data in grouped_spikes.items():
        neuron_types = session_info['Type']
        neurons = spike_data.columns
        # Limit data to time window and drop NaNs
        limited_spike_data = spike_data.apply(lambda x: x[(x <= time_limit)].dropna())
        for neuron in neurons:
            spikes = limited_spike_data[neuron]
            spikes = spikes[~np.isnan(spikes)]
            num_spikes = len(spikes)
            firing_rate = num_spikes / time_limit  # spikes per second
            neuron_type = neuron_types.get(neuron, 'Unknown')
            firing_rates.append({
                'Neuron': neuron,
                'Group': group,
                'FiringRate': firing_rate,
                'Type': neuron_type
            })
    return pd.DataFrame(firing_rates)

firing_rates_df = compute_firing_rates(grouped_spikes, session_info, time_limit=10)
def plot_firing_rates(firing_rates_df):
    plt.figure(figsize=(8, 6))
    sns.swarmplot(data=firing_rates_df, x='Type', y='FiringRate')
    sns.pointplot(
        data=firing_rates_df, x='Type', y='FiringRate',
        estimator=np.mean, ci='sd', markers='D', color='red',
        errwidth=1.5, capsize=0.1, join=False
    )

    da_firing_rates = firing_rates_df[firing_rates_df['Type'] == 'DA']['FiringRate']
    non_da_firing_rates = firing_rates_df[firing_rates_df['Type'] == 'non-DA']['FiringRate']

    da_mean = da_firing_rates.mean()
    da_std = da_firing_rates.std()
    non_da_mean = non_da_firing_rates.mean()
    non_da_std = non_da_firing_rates.std()

    # Print out the mean firing rates
    print(f"DA neurons mean firing rate: {da_mean:.2f} ± {da_std:.2f} spikes/s")
    print(f"non-DA neurons mean firing rate: {non_da_mean:.2f} ± {non_da_std:.2f} spikes/s")

    # Add text annotations for the means on the plot
    # Determine y-positions for annotations
    da_y_pos = da_mean + da_std + 0.05 * firing_rates_df['FiringRate'].max()
    non_da_y_pos = non_da_mean + non_da_std + 0.05 * firing_rates_df['FiringRate'].max()

    plt.text(
        x=0, y=da_y_pos,
        s=f"Mean: {non_da_mean:.2f} ± {da_std:.2f} Hz",
        color='red', ha='center', va='bottom', fontsize=10
    )
    plt.text(
        x=1, y=non_da_y_pos,
        s=f"Mean: {da_mean:.2f} ± {non_da_std:.2f} Hz",
        color='red', ha='center', va='bottom', fontsize=10
    )

    plt.tight_layout()
    plt.show()
# Plot firing rates
plot_firing_rates(firing_rates_df)



def plot_firing_rates(firing_rates_df):
    plt.figure(figsize=(8, 6))
    sns.swarmplot(data=firing_rates_df, x='Type', y='FiringRate')
    plt.title('Distribution of Firing Rates by Neuron Type')
    plt.xlabel('Neuron Type')
    plt.ylabel('Firing Rate (spikes/s)')
    plt.show()

# Plotting the swarm and violin plots with shuffled data
def clean_sttc_values(val):
    if isinstance(val, list):
        return np.nan
    else:
        return val

print(df_plot.head(50))
df_plot['STTC'] = df_plot['STTC'].apply(clean_sttc_values)

# Remove entries with NaN STTC values
df_plot = df_plot.dropna(subset=['STTC'])

# Ensure that 'STTC' is of type float
df_plot['STTC'] = df_plot['STTC'].astype(float)

# Proceed to plotting
def plot_synchrony_results(df_plot):
    plt.figure(figsize=(16, 8))

    # Swarm plot
    sns.swarmplot(data=df_plot, x='Category', y='STTC', hue='Shuffle', dodge=True)
    plt.xticks(rotation=45)
    plt.title('Swarm Plot of STTC Synchrony Measures')
    plt.xlabel('Category')
    plt.ylabel('STTC')
    plt.legend(title='Shuffle', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_synchrony_results(df_plot)

def perform_statistical_tests(df_plot):
    categories = df_plot['Category'].unique()
    p_values = {}
    for category in categories:
        df_cat = df_plot[df_plot['Category'] == category]
        # Ensure the data is paired by sorting by 'Group'
        df_original = df_cat[df_cat['Shuffle'] == 'Original'].sort_values('Group')
        df_shuffled = df_cat[df_cat['Shuffle'] == 'Shuffled'].sort_values('Group')
        original = df_original['STTC'].values
        shuffled = df_shuffled['STTC'].values
        # Perform the Wilcoxon signed-rank test
        if len(original) == len(shuffled) and len(original) > 0:
            try:
                stat, p_value = wilcoxon(original, shuffled)
                p_values[category] = p_value
            except ValueError:
                p_values[category] = np.nan
        else:
            p_values[category] = np.nan
    return p_values

def plot_event_plot(da_neurons, spike_data, group_str, sttc_values):
    plt.figure(figsize=(10, len(da_neurons) * 0.5))
    for idx, neuron in enumerate(da_neurons):
        spikes = spike_data[neuron]
        spikes = np.asarray(spikes)
        spikes = spikes[~np.isnan(spikes)]
        plt.eventplot(spikes, lineoffsets=idx + 1, linelengths=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron')
    plt.title(f'Group: {group_str} - DA Neurons\nSTTC: {sttc_values.get("DA_all_STTC", "N/A"):.3f}')
    plt.yticks(range(1, len(da_neurons) + 1), da_neurons)
    plt.tight_layout()
    plt.show()

p_values = perform_statistical_tests(df_plot)
print("P-values from Wilcoxon signed-rank test:")
for category, p_value in p_values.items():
    print(f"{category}: p = {p_value}")


