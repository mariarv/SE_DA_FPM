########################################
# Simple network simulation : pattterns of firing + opto template
# bursting and tonic firing
#######################################

import numpy as np
import matplotlib.pyplot as plt
import metrics_analysis as m_a
import pandas as pd
import os
from multiprocessing import Pool

# Ensure the directory for saving plots exists
output_dir = "results/plots/firing_patterns"
data_output_dir = "results/generated_firing_patterns"

# Define the different firing types
firing_types = ['synchronous', 'asynchronous', 'random']

def plot_raster(spike_trains, dt, max_time_ms=30000):
    num_neurons = len(spike_trains)

    num_neurons = len(spike_trains)
    
    # Create a new figure for the raster plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Loop over each neuron and plot its spike times
    for neuron_idx, spike_train in enumerate(spike_trains):
        spike_times = np.where(spike_train > 0)[0] * dt  # Extract spike times based on non-zero entries
        ax.scatter(spike_times, np.ones_like(spike_times) * neuron_idx, s=2, color="black")

    ax.set_title('Raster Plot of Neuronal Spiking Activity')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.set_ylim(-0.5, num_neurons - 0.5)

    plt.show()

def simulate_combination(args):
    tonic_firing_type, bursting_firing_type, template, time_vector, fs, nperseg, noverlap, max_freq = args
    
    all_power_spectra_sim = []  # To store the power spectra for each repetition (simulated)
    
    for repeat in range(repeats):
        print(f"Simulating for Tonic: {tonic_firing_type}, Bursting: {bursting_firing_type}, Repeat {repeat + 1}/{repeats}")

        # Generate tonic and bursting spike trains
        tonic_spike_trains = m_a.generate_tonic_spike_trains(
            num_neurons=num_neurons_tonic,
            total_time=total_time,
            dt=dt,
            tonic_rate=tonic_rate,
            tonic_interval_noise=tonic_interval_noise,
            firing_type=tonic_firing_type,
            min_isi_time_points=150
        )

        bursting_spike_trains = m_a.generate_tonic_to_bursting_spike_trains(
            num_neurons=num_neurons_bursting,
            total_time=total_time,
            dt=dt,
            tonic_rate=2,
            burst_rate=burst_rate,
            inter_spike_interval=inter_spike_interval,
            switch_prob=switch_prob,
            firing_type=bursting_firing_type,
            avg_spikes_per_burst=spikes_per_burst,
            tonic_interval_noise=tonic_interval_noise,
            sd_spikes_per_burst=3,
            jitter=jitter
        )

        # Combine spike trains
        all_spike_trains = np.array(tonic_spike_trains + bursting_spike_trains)

        # Generate calcium traces by convolving spike trains with the template
        calcium_traces = []
        for spike_train in all_spike_trains:
            calcium_trace = np.convolve(spike_train, template, mode='full')[:time_points]
            calcium_traces.append(calcium_trace)
        calcium_traces = np.array(calcium_traces)

        # Calculate the bulk ΔF/F (sum across all neurons)
        bulk_calcium_trace = np.sum(calcium_traces, axis=0)

        # Save the bulk calcium trace to CSV
        bulk_trace_df = pd.DataFrame({
            'Time (s)': time_vector,
            'Bulk ΔF/F': bulk_calcium_trace
        })
        bulk_trace_df.to_csv(f"{data_output_dir}/bulk_trace_tonic_{tonic_firing_type}_bursting_{bursting_firing_type}_repeat_{repeat + 1}.csv", index=False)

        # Plot and save raster plot (only first 30 seconds)
        plt.figure(figsize=(10, 8))
        plot_raster(all_spike_trains, dt, max_time_ms=30000)
        plt.savefig(f"{output_dir}/raster_tonic_{tonic_firing_type}_bursting_{bursting_firing_type}_repeat_{repeat + 1}.pdf")
        plt.close()

        # Plot and save bulk calcium trace
        plt.figure(figsize=(12, 6))
        plt.plot(time_vector, bulk_calcium_trace, label='Bulk ΔF/F')
        plt.xlabel('Time (s)')
        plt.ylabel('ΔF/F (a.u.)')
        plt.title(f'Bulk Calcium Trace for Tonic: {tonic_firing_type}, Bursting: {bursting_firing_type} (Repeat {repeat + 1})')
        plt.legend()
        plt.savefig(f"{output_dir}/bulk_calcium_tonic_{tonic_firing_type}_bursting_{bursting_firing_type}_repeat_{repeat + 1}.pdf")
        plt.close()

        # Compute and store power spectrum
        bulk_calcium_trace_ = m_a.robust_zscore(bulk_calcium_trace[1000:])
        freqs_sim, power_dB_sim = m_a.compute_power_spectrum_dB(
            bulk_calcium_trace_, fs, nperseg=nperseg, noverlap=noverlap, max_freq=max_freq
        )
        all_power_spectra_sim.append(power_dB_sim)

    # After all repetitions, plot and save power spectra
    plt.figure(figsize=(10, 4))

    # Plot the mean simulated spectrum across repetitions
    mean_power_spectrum_sim = np.mean(all_power_spectra_sim, axis=0)
    plt.plot(freqs_sim, mean_power_spectrum_sim, color='blue', label='Mean Simulated Power Spectrum')
    plt.title(f'Power Spectrum for Tonic: {tonic_firing_type}, Bursting: {bursting_firing_type}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.legend()
    plt.savefig(f"{output_dir}/power_spectrum_tonic_{tonic_firing_type}_bursting_{bursting_firing_type}.pdf")
    plt.close()

    return f"Finished simulation for Tonic: {tonic_firing_type}, Bursting: {bursting_firing_type}"

# Function to parallelize over firing type combinations
def run_simulations_parallel(num_workers=4):
    all_args = []
    
    for tonic_firing_type in firing_types:
        for bursting_firing_type in firing_types:
            args = (
                tonic_firing_type, 
                bursting_firing_type, 
                template, 
                time_vector, 
                fs, 
                nperseg, 
                noverlap, 
                max_freq
            )
            all_args.append(args)

    # Use Pool to parallelize the process with a specified number of workers
    with Pool(processes=num_workers) as pool:
        results = pool.map(simulate_combination, all_args)
    
    for result in results:
        print(result)


num_neurons_tonic = 60
num_neurons_bursting = 40
total_time = 60  # Total time in seconds
dt = 0.001  # Time step in seconds (1 ms)
tonic_rate = 4  # Firing rate for tonic neurons (3 Hz)
burst_rate = 3  # Bursts per second for bursting neurons (0.5 Hz or 1 burst every 2 seconds)
spikes_per_burst = 15  # Number of spikes per burst
inter_spike_interval = 0.005  # Interval between spikes within a burst (10 ms)
tonic_interval_noise = 0.1  # Noise as a percentage of the interval
time_points = int(total_time / dt)
switch_prob = 0.1
jitter = 0.05

# Create time vector
time_vector = np.arange(0, total_time, dt)

# Load or define the template waveform for a single spike
TEMPLATE_FILE_PATH = '/Users/reva/Documents/Python/SE_DA_FPM/data/Templates/Before_Cocaine_DS_mean_traces_dff.csv'
template_ = m_a.load_template(TEMPLATE_FILE_PATH, "25")
template = template_[450:]
extension = np.full((6000 - len(template),), template[2050])
template = np.concatenate([template, extension])

# If you need to ensure that extended_template has exactly 6000 elements
template = template[:6000]
template_length = len(template)

fs = 1000
nperseg = 4096
noverlap = 3072
max_freq = 30
repeats = 10


if __name__ == '__main__':
    # Specify the number of workers here
    num_workers = 1  # Set the number of parallel processes (workers)

    # Run the simulations in parallel with the specified number of workers
    run_simulations_parallel(num_workers=num_workers)