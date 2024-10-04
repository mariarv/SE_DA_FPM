#read and plot the raster of VTA DA neurons firing, data from the Kremer et al. 2020

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import metrics_analysis as m_a
from mpl_toolkits.mplot3d import Axes3D

def generate_spike_trains(modality="distribution", num_neurons=100, duration=60, fs=1000, precomputed_pdfs=None, valid_neurons_list=None, poisson_rate=5, reference_spike_train=None, jitter_std=0.002, repetitions=5):
    """
    Generate spike trains based on different modalities:
    - "distribution": Sample spike times from a given PDF distribution.
    - "poisson": Generate Poisson-distributed spike trains.
    - "repeated": Replicate a reference spike train with and without jitter.
    
    Parameters:
    - modality: Type of spike train generation ("distribution", "poisson", "repeated").
    - num_neurons: Number of neurons to generate spike trains for.
    - duration: Duration of spike train in seconds.
    - fs: Sampling frequency (used for Poisson generation).
    - pdf: Probability density function for "distribution" modality.
    - x_values: x-values corresponding to the PDF for sampling.
    - poisson_rate: Firing rate for Poisson-distributed spikes (in Hz).
    - reference_spike_train: Reference spike train to be replicated in "repeated" modality.
    - jitter_std: Standard deviation of Gaussian jitter for introducing variations in "repeated" modality.
    - repetitions: Number of neurons to exactly replicate the reference spike train in "repeated" modality.

    Returns:
    - spike_trains: List of spike trains for each neuron.
    """
    spike_trains = []

    if modality == "distribution":
        # Check if PDF and x_values are provided
        if precomputed_pdfs is None or valid_neurons_list is None:
            raise ValueError("PDF and x_values must be provided for 'distribution' modality.")

        
        # Sample spike times based on the PDF
        for _ in range(num_neurons):
            i = np.random.randint(0, 2)
            neuron=valid_neurons_list[i]
            precomputed_data = precomputed_pdfs[neuron]
            x_values = precomputed_data["x_values"]
            spiking_times = m_a.sample_spiking_for_duration(precomputed_data['normalized_combined_pdf'], x_values, duration=duration)
            spike_trains.append(spiking_times)

    elif modality == "poisson":
        # Generate Poisson spike trains for each neuron
        dt = 1 / fs  # Time step
        time_points = int(duration / dt)
        for _ in range(num_neurons):
            # Generate spike times based on Poisson distribution
            spike_train_binary = np.random.rand(time_points) < (poisson_rate * dt)
            spiking_times = np.where(spike_train_binary)[0] * dt
            spike_trains.append(spiking_times)

    elif modality == "repeated":
        # Ensure reference spike train is provided
        if reference_spike_train is None:
            raise ValueError("Reference spike train must be provided for 'repeated' modality.")
        

        # Generate repeated spike trains with and without jitter
        for _ in range(repetitions):
                # Introduce a small jitter to the replicated spike train
            jitter = np.random.normal(0, jitter_std, size=len(reference_spike_train))
            spiking_times = reference_spike_train + jitter
            spiking_times = np.clip(spiking_times, 0, duration)  # Ensure times are within bounds
            spiking_times = np.sort(spiking_times)  # Ensure spiking times are sorted

            spike_trains.append(spiking_times)

        for _ in range(num_neurons-repetitions):
            # Generate an independent spike train for this neuron
            i = np.random.randint(0, 5)
            neuron=valid_neurons_list[i]
            precomputed_data = precomputed_pdfs[neuron]
            x_values = precomputed_data["x_values"]
            spiking_times = m_a.sample_spiking_for_duration(precomputed_data['normalized_combined_pdf'], x_values, duration=duration)

            spike_trains.append(spiking_times)

    else:
        raise ValueError("Invalid modality. Choose from 'distribution', 'poisson', or 'repeated'.")

    return spike_trains


def insert_synchronized_bursts(spike_trains, burst_times, burst_duration=0.05):
    """
    Insert synchronized bursts at specific times into the given spike trains.

    Parameters:
    - spike_trains: List of spike trains (each train is an array of spike times).
    - burst_times: List of times at which to insert synchronized bursts.
    - burst_duration: Duration of each synchronized burst (default: 0.05 seconds).

    Returns:
    - synchronized_spike_trains: List of spike trains with synchronized bursts added.
    """

    compensated_spike_trains = spike_trains[:]

    for neuron_idx in range(25):
        # Get the spike train for the current neuron
        spike_train = compensated_spike_trains[neuron_idx]
        # Compensate by removing some natural bursts from the original spike train
        spike_train = find_and_replace_natural_bursts(spike_train, burst_isi_threshold=0.02, num_bursts_to_replace=len(burst_times))

        # Insert synchronized bursts into the modified spike train
        for burst_time in burst_times:
            # Sample a burst duration around the average with some variability
            burst_duration = np.abs(np.random.normal(loc=burst_duration, scale=0.05))
            num_spikes=np.random.randint(3, 7)
            # Create a burst of spikes around the burst_time using the sampled burst duration
            burst_spikes = np.arange(burst_time, burst_time + burst_duration, burst_duration / num_spikes)  # 5 spikes per burst
            # Insert burst spikes into the spike train
            spike_train = np.concatenate([spike_train, burst_spikes])
            # Sort to maintain the temporal order
            compensated_spike_trains[neuron_idx] = np.sort(spike_train)

    return compensated_spike_trains


def find_and_replace_natural_bursts(spike_train, burst_isi_threshold=0.05, num_bursts_to_replace=1):
    """
    Find and replace natural bursts in a spike train by a single spike at the burst onset.

    Parameters:
    - spike_train: Array of spike times for a single neuron.
    - burst_isi_threshold: ISI threshold (in seconds) to define a burst.
    - num_bursts_to_replace: Number of bursts to replace with a single spike.

    Returns:
    - modified_spike_train: Modified spike train with bursts replaced by a single spike.
    """
    # Calculate ISIs to identify bursts
    isis = np.diff(spike_train)
    burst_indices = np.where(isis < burst_isi_threshold)[0]
    
    # Group burst indices into separate bursts
    burst_segments = np.split(burst_indices, np.where(np.diff(burst_indices) > 1)[0] + 1)
    
    # Ensure we do not exceed the available number of natural bursts
    num_bursts_to_replace = min(len(burst_segments), num_bursts_to_replace)
    
    # Replace bursts with single spikes
    for segment in burst_segments[:num_bursts_to_replace]:
        if len(segment) == 0:
            continue
        # Remove all spikes in the burst except the first one
        burst_start_index = segment[0]
        burst_end_index = segment[-1] + 1  # End index of the burst (inclusive)
        
        # Ensure the indices are within the valid range of the spike train
        if burst_end_index < len(spike_train):
            spike_train = np.delete(spike_train, range(burst_start_index + 1, burst_end_index + 1))
    
    return spike_train

file_name = '/Users/reva/Documents/Neuron_SpikeTimes_BeforeCue_Concatenated.xlsx'
df = pd.read_excel(file_name)

file_path = '/Users/reva/Documents/Neuron_SpikeTimes_BeforeCue_Concatenated.xlsx'
total_time = 60  # Total time in seconds
dt = 0.001  # Time step in seconds (1 ms)
time_points = int(total_time / dt)
# Step 2: Generate multiple spike train patterns using the precomputed PDFs
all_spike_trains_list = []  # List to store all patterns
# Load or define the template waveform for a single spike
TEMPLATE_FILE_PATH = '/Users/reva/Documents/Python/SE_DA_FPM/data/Templates/Before_Cocaine_VS_mean_traces_dff.csv'
template_ = m_a.load_template(TEMPLATE_FILE_PATH, "25")
template = template_[400:]
extension = np.full((6000 - len(template),), template[2050])
template = np.concatenate([template, extension])
template = (template[:6000]-  template[0])

template_length = len(template)

fs = 1000
nperseg = 4096
noverlap = 3072
max_freq = 30
real_data_spectra_path = "data/combined_VS_before_spectra.csv"
real_data_df = pd.read_csv(real_data_spectra_path, skiprows=1, index_col=0)
all_power_spectra_sim=[]
firing_rates_sim=[]
# Read ISI data
isi_data_full = pd.read_excel(file_path)
valid_neurons_list = [col for col in isi_data_full.columns if not isi_data_full[col].dropna().empty]  # Filter valid neurons
bins = 1000  # Number of bins for histograms
cell_distributions_and_spiking = {}
all_spike_trains = []c
precomputed_pdfs = {}

for neuron in valid_neurons_list:
    # Extract ISI data for the neuron
    spike_times = isi_data_full[neuron].dropna().values  # Extract ISI data
    neuron_isis = np.diff(spike_times)
    neuron_isis = neuron_isis[neuron_isis > 0]  # Remove non-positive ISIs

    # Calculate the histogram and bin centers (only need to do this once)
    hist_values, bin_edges = np.histogram(neuron_isis, bins=bins, density=True)
    x_values = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Precompute the combined PDF and individual components once
    normalized_combined_pdf, scaled_burst_pdf_matched, scaled_tonic_pdf_matched = m_a.extract_combined_distribution_for_cell(neuron_isis, x_values)
    
    # Store the precomputed values in a dictionary for reuse
    precomputed_pdfs[neuron] = {
        'normalized_combined_pdf': normalized_combined_pdf,
        'x_values': x_values
    }

modality_choice = "distribution"  # Options: "distribution", "poisson", "repeated"
num_neurons = 100  # Total number of neurons
duration = 60  # Duration in seconds
fs = 1000  # Sampling frequency



# For "distribution" modality, use the precomputed PDFs and x_values
pdf = None
x_values = None
num_patterns = 20  # Number of patterns to generate
burst_times = np.arange(0, duration, 5)


for pattern_idx in range(num_patterns):
    i = np.random.randint(0, 2)
    n = np.random.randint(15, 30)

    reference_spike_train = m_a.sample_spiking_for_duration(precomputed_pdfs[valid_neurons_list[i] ]['normalized_combined_pdf'], precomputed_pdfs[neuron]['x_values'], duration=duration)
    if modality_choice == "distribution" or modality_choice == "repeated" :
        neuron = valid_neurons_list[i]  # Choose the first neuron as an example
        precomputed_data = precomputed_pdfs[neuron]
        pdf = precomputed_data['normalized_combined_pdf']
        x_values = precomputed_data['x_values']
    # Generate spike trains
    all_spike_trains = generate_spike_trains(
        modality=modality_choice,
        num_neurons=num_neurons,
        duration=duration,
        fs=fs,
        precomputed_pdfs=precomputed_pdfs,
        valid_neurons_list=valid_neurons_list,
        poisson_rate=5,
        reference_spike_train=reference_spike_train,
        jitter_std=0.01,
        repetitions=n  # Number of repeated spike trains
    )

    # Add synchrony if needed
    #all_spike_trains = synchronize_bursts(all_spike_trains_, fraction_to_sync=0.5, burst_isi_threshold=0.05, duration=duration)
    burst_times = []
    current_time = 0
    while current_time < duration:
        # Sample a random interval between bursts within the specified range
        interval = np.random.randint(3, 7)  # Random interval between bursts in seconds
        current_time += interval
        if current_time < duration:
            burst_times.append(current_time)
    # Insert synchronized bursts into the Poisson spike trains
    #all_spike_trains = insert_synchronized_bursts(all_spike_trains_, burst_times, burst_duration=0.1)



    calcium_traces = []
    for spike_times in all_spike_trains:
        # Step 1: Convert spike times into a binary array
        spike_train_binary = np.zeros(time_points)
        spike_indices = np.floor(spike_times / dt).astype(int)  # Convert spike times to indices based on dt
        spike_indices = spike_indices[spike_indices < time_points]  # Ensure indices are within bounds
        spike_train_binary[spike_indices] = 1  # Set spikes in the binary array

        # Step 2: Convolve spike train binary with template to generate calcium trace
        calcium_trace = np.convolve(spike_train_binary, template, mode='full')[:time_points]
        calcium_traces.append(calcium_trace)

    min_length = min(len(trace) for trace in calcium_traces)
    truncated_calcium_traces = [trace[:min_length] for trace in calcium_traces]

    # Step 3: Replace the original calcium_traces with the truncated versions
    calcium_traces = truncated_calcium_traces

    calcium_traces = np.array(calcium_traces)
    bulk_calcium_trace = np.sum(calcium_traces, axis=0)
    #plt.savefig(f"{output_dir}/bulk_calcium_tonic_{tonic_firing_type}_bursting_{bursting_firing_type}_repeat_{repeat + 1}.pdf")
    # Compute the power spectrum for this simulation
    print(np.mean(bulk_calcium_trace[2000:]))
    bulk_calcium_trace_ =m_a.robust_zscore(bulk_calcium_trace[2000:]) # Z-score normalization
    freqs_sim, power_dB_sim = m_a.compute_power_spectrum_dB(
        bulk_calcium_trace_, fs, nperseg=nperseg, noverlap=noverlap, max_freq=max_freq
    )
    all_power_spectra_sim.append(power_dB_sim)
    
    H_before = m_a.create_sliding_windows(bulk_calcium_trace_, 10000, 200)
    H_before_flat = H_before.reshape(-1, H_before.shape[1])
    U, S, Vt = np.linalg.svd(H_before_flat, full_matrices=False)
    num_modes = 3  # Number of modes to retain
    umap_before = U
    time=np.arange(len(bulk_calcium_trace_)) / fs

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])  # 2 rows, 1 column; adjust ratios as needed

    # First subplot: 3D plot
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax1.plot(umap_before[:, 0], umap_before[:, 1], umap_before[:, 2], marker='o', markersize=1, linestyle='-', color='blue')
    ax1.set_title("3D UMAP Plot")
    ax1.set_xlabel('UMAP Dimension 1')
    ax1.set_ylabel('UMAP Dimension 2')
    ax1.set_zlabel('UMAP Dimension 3')

    # Second subplot: Calcium trace
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time,bulk_calcium_trace_, color='orange')
    ax2.set_title("Bulk DA Trace")
    ax2.set_xlabel('Time(s)')
    ax2.set_ylabel('Normalized')
     
    # Adjust layout to prevent overlap
    plt.savefig(f'results/plots/POD_plots/VS_POD_{pattern_idx}_simulated_no_sync.pdf')
    #plt.show()
    fig = plt.figure(figsize=(10, 8))
    plt.plot(time, bulk_calcium_trace_)
    plt.ylabel("AUC (robust z-score)")
    plt.xlabel("Time(s)")
    plt.savefig(f'results/plots/DA_relese_{pattern_idx}_simulated_no_sync.pdf')


    # Create a new figure for the raster plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Loop over each neuron and plot its spike times
    for neuron_idx, spike_train in enumerate(all_spike_trains):
        ax.scatter(spike_train, np.ones_like(spike_train) * neuron_idx, s=2, color="black")
    for burst_time in burst_times:
        ax.axvline(burst_time, color='red', linestyle='--', linewidth=1, label='Burst Time' if burst_time == burst_times[0] else "")

    ax.set_title('Raster Plot of Neuronal Spiking Activity')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.set_ylim(-0.5, 100 - 0.5)
    ax.set_xlim(0, 30)
    plt.xlabel('Time (s)')
    plt.ylabel('Spike Events')
    plt.legend()
    plt.grid()
    plt.savefig(f'results/plots/Patterns_{pattern_idx}_simulated_no_sync.pdf')

real_spectra = []
for i in range(real_data_df.shape[0]):
    bulk_calcium_trace = real_data_df.iloc[i].values  # Get the calcium trace for neuron i
    freqs, power_dB = m_a.compute_power_spectrum_dB(bulk_calcium_trace, fs, nperseg=nperseg, noverlap=noverlap, max_freq=max_freq)
    real_spectra.append((freqs, power_dB))
        
plt.figure(figsize=(10, 6))
mean_power_spectrum_sim = np.mean(all_power_spectra_sim, axis=0)

all_power_spectra_exp=[]
# Plot individual experimental spectra
for freqs_exp, power_dB_exp in real_spectra:
    all_power_spectra_exp.append(power_dB_exp)
    plt.plot(freqs_exp, power_dB_exp, color='grey', alpha=0.5)

# Plot individual simulated spectra
for i, power_dB_sim in enumerate(all_power_spectra_sim):
    plt.plot(freqs_sim, power_dB_sim, color='green', alpha=0.3)

# Plot mean of simulated spectra
plt.plot(freqs_sim, mean_power_spectrum_sim, color='green', linewidth=2)

# Compute mean of experimental spectra and plot
mean_power_spectrum_exp = np.nanmean(all_power_spectra_exp, axis=0)
plt.plot(freqs_exp, mean_power_spectrum_exp, color='black', linewidth=4)
plt.savefig(f'results/plots/Spectra_simulated_no_sync.pdf')



v_before = np.gradient(bulk_calcium_trace_, time_points)
fig = plt.figure(figsize=(14, 12))
plt.plot(bulk_calcium_trace_, v_before,  marker='o', markersize=1, linestyle='-', color='blue')       
