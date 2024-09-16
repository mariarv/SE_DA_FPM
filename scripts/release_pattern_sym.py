########################################
# Simple network simulation : pattterns of firing + opto template
# bursting and tonic firing
#######################################

import numpy as np
import matplotlib.pyplot as plt
import metrics_analysis as m_a
import pandas as pd
import os

# Ensure the directory for saving plots exists
output_dir = "results/plots/firing_patterns"
data_output_dir = "results/data/firing_patterns"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(data_output_dir):
    os.makedirs(data_output_dir)
# Define the different firing types
firing_types = [ 'synchronous','asynchronous', 'random']

def plot_raster(spike_trains, dt):
    """
    spike_trains: A 2D numpy array where each row represents the spike train of a neuron.
                  Each element is binary (1 for spike, 0 for no spike) or a list of spike timestamps.
    dt: Time step used for the simulation in ms or time units.
    """
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

def plot_calcium_and_spikes(calcium_traces, spike_trains, dt, neuron_idx, start_idx=1000):    
    """
    Plots the calcium trace from the DataFrame and corresponding spike train for a specific neuron.
    
    Parameters:
    ground_truth_df (DataFrame): Contains the calcium traces in the 'Smoothed Bulk ΔF/F' column.
    spike_trains (2D numpy array): Spike trains for all neurons (binary, 1 for spike).
    dt (float): Time step between samples.
    neuron_idx (int): Index of the neuron to plot.
    start_idx (int): Starting index to plot the calcium traces and spike trains (default: 1000).
    """
    # Extract the specific neuron's calcium trace from the DataFrame starting at `start_idx`
    calcium_trace = ground_truth_df['Smoothed Bulk ΔF/F'].values
    
    # Extract the corresponding spike train for this neuron
    spike_train = spike_trains[neuron_idx][start_idx:]
    
    # Create the time axis based on dt and the length of the calcium trace
    time_axis = np.arange(len(calcium_trace)) * dt
    
    # Plot the calcium trace
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_axis, calcium_trace, label=f'Neuron {neuron_idx} Calcium Trace', color='blue')
    
    # Get the spike times (where the spike train is non-zero)
    spike_times = np.where(spike_train > 0)[0] * dt
    
    # Plot the spike train as dots on top of the calcium trace
    ax.scatter(spike_times, calcium_trace[spike_times.astype(int)], color='red', marker='o', label='Spike Events')

    # Customize the plot
    ax.set_title(f'Calcium Trace and Spikes for Neuron {neuron_idx}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Calcium Signal (ΔF/F)')
    ax.legend(loc='upper right')
    plt.show()

def generate_tonic_spike_trains(num_neurons, total_time, dt, tonic_rate, tonic_interval_noise, firing_type, min_isi_time_points):
    """
    Generate spike trains for tonic neurons with different firing rates drawn from a uniform distribution.

    Args:
        num_neurons (int): Number of neurons.
        total_time (float): Total simulation time in seconds.
        dt (float): Time step in seconds.
        tonic_interval_noise (float): Noise as a percentage of the interval.
        firing_type (str): Type of tonic firing ("random", "synchronous", "asynchronous").
        min_isi_time_points (int): Minimum interval between spikes (in time points).

    Returns:
        list: List of spike trains for the tonic neurons.
    """
    time_points = int(total_time / dt)

    # Define variable firing rates from 0.5 Hz to 8 Hz
    firing_rates = np.random.uniform(0.5, 6, num_neurons)

    # Adjust the firing rates to ensure the mean is exactly 4 Hz
    current_mean = np.mean(firing_rates)
    target_mean = 3.0
    firing_rates = firing_rates + (target_mean - current_mean)
    
    spike_trains = []
    jitter=0.05            
    max_firing_rate = 6.0  # Maximum firing rate used for the base train
    max_interval_between_spikes = int(1 / max_firing_rate / dt)
    base_spike_train = np.zeros(time_points)
    current_time = 0

    # Generate the base spike train with the highest firing frequency (8 Hz)
    while current_time < time_points:
        current_time += max_interval_between_spikes
        if current_time < time_points:
            base_spike_train[current_time] = 1

    for i in range(num_neurons):
        # Determine the mean interval between spikes for each neuron
        tonic_rate = firing_rates[i]
        mean_interval_between_spikes = int(1 / tonic_rate / dt)  # Mean time points between spikes
        
        if firing_type == "synchronous":
            # Synchronous firing pattern: Generate base spike train with some noise

            # Create spike trains for each neuron by removing spikes equidistantly based on their firing rate
            max_jitter_points = int(jitter / dt)  # Convert jitter from seconds to time points

            spike_train = np.zeros(time_points)

            # Calculate the interval for keeping spikes equidistantly based on the neuron’s firing rate
            keep_every_nth_spike = int(max_firing_rate / tonic_rate)
            if keep_every_nth_spike ==0 :
                continue
            
            spike_indices = np.where(base_spike_train > 0)[0]

            for idx, spike_time in enumerate(spike_indices):
                # Keep the spike if it's one of the every-nth spikes
                if idx % keep_every_nth_spike == 0:
                    # Introduce consistent jitter
                    jittered_time = spike_time + np.random.randint(-max_jitter_points, max_jitter_points)
                    jittered_time = min(max(jittered_time, 0), time_points - 1)  # Ensure within bounds
                    spike_train[jittered_time] = 1

            spike_trains.append(spike_train)


        elif firing_type == "asynchronous":
            # Asynchronous firing: Each neuron fires independently with variable intervals
            spike_train = np.zeros(time_points)
            current_time = 0
            while current_time < time_points:
                noisy_interval = mean_interval_between_spikes + int(
                    np.random.normal(0, tonic_interval_noise * mean_interval_between_spikes)
                )
                current_time += max(1, noisy_interval)
                if current_time < time_points:
                    spike_train[current_time] = 1
            spike_trains.append(spike_train)

        elif firing_type == "random":
            # Random firing: Each neuron generates spikes with a Poisson process, varying firing rates
            spike_train = np.zeros(time_points)
            spike_prob = tonic_rate * dt  # Probability of a spike per time step
            spikes = np.random.rand(time_points) < spike_prob
            
            # Enforce minimum ISI by post-processing
            spike_times = np.where(spikes)[0]
            valid_spike_times = []
            if len(spike_times) > 0:
                valid_spike_times.append(spike_times[0])
                for spike_time in spike_times[1:]:
                    if spike_time - valid_spike_times[-1] >= min_isi_time_points:
                        valid_spike_times.append(spike_time)
            spike_train[valid_spike_times] = 1
            spike_trains.append(spike_train)

        else:
            raise ValueError("Invalid firing type. Choose 'random', 'synchronous', or 'asynchronous'.")
    
    return spike_trains

def generate_bursting_spike_trains(num_neurons, total_time, dt, burst_rate, inter_spike_interval, firing_type, avg_spikes_per_burst, sd_spikes_per_burst, jitter):
    """
    Generate spike trains for bursting neurons with different firing types.

    Args:
        num_neurons (int): Number of neurons.
        total_time (float): Total simulation time in seconds.
        dt (float): Time step in seconds.
        burst_rate (float): Bursts per second.
        inter_spike_interval (float): Interval between spikes within a burst (in seconds).
        firing_type (str): Type of bursting firing ("random", "synchronous", "asynchronous").
        avg_spikes_per_burst (int): Average number of spikes per burst.
        sd_spikes_per_burst (int): Standard deviation of spikes per burst.
        jitter (float): Maximum jitter (in seconds) to apply for "synchronous" firing.

    Returns:
        list: List of spike trains for the bursting neurons.
    """
    time_points = int(total_time / dt)

    if firing_type == "synchronous":
        # Generate base burst train
        base_spike_train = np.zeros(time_points)
        burst_times = np.random.poisson(burst_rate * dt, time_points)
        burst_indices = np.where(burst_times > 0)[0]
        for burst_start in burst_indices:
            spikes_in_burst = max(1, int(np.random.normal(avg_spikes_per_burst, sd_spikes_per_burst)))
            for j in range(spikes_in_burst):
                spike_time = burst_start + int(j * inter_spike_interval / dt)
                if spike_time < time_points:
                    base_spike_train[spike_time] = 1

        # Create spike trains with slight jitter
        spike_trains = []
        max_jitter_points = int(jitter / dt)  # Convert jitter from seconds to time points
        for i in range(num_neurons):
            spike_train = np.zeros(time_points)
            spike_indices = np.where(base_spike_train > 0)[0]
            for spike_time in spike_indices:
                jittered_time = spike_time + np.random.randint(-max_jitter_points, max_jitter_points)
                jittered_time = min(max(jittered_time, 0), time_points - 1)  # Ensure within bounds
                spike_train[jittered_time] = 1
            spike_trains.append(spike_train)

    elif firing_type == "asynchronous":
        spike_trains = []
        for i in range(num_neurons):
            spike_train = np.zeros(time_points)
            burst_times = np.random.poisson(burst_rate * dt, time_points)
            burst_indices = np.where(burst_times > 0)[0]
            for burst_start in burst_indices:
                spikes_in_burst = max(1, int(np.random.normal(avg_spikes_per_burst, sd_spikes_per_burst)))
                for j in range(spikes_in_burst):
                    spike_time = burst_start + int(j * inter_spike_interval / dt)
                    if spike_time < time_points:
                        spike_train[spike_time] = 1
            spike_trains.append(spike_train)

    elif firing_type == "random":
        spike_trains = []
        for i in range(num_neurons):
            spike_train = np.zeros(time_points)
            for t in range(time_points):
                if np.random.rand() < burst_rate * dt:
                    spikes_in_burst = max(1, int(np.random.normal(avg_spikes_per_burst, sd_spikes_per_burst)))
                    for j in range(spikes_in_burst):
                        spike_time = t + int(j * inter_spike_interval / dt)
                        if spike_time < time_points:
                            spike_train[spike_time] = 1
            spike_trains.append(spike_train)
    
    else:
        raise ValueError("Invalid firing type. Choose 'random', 'synchronous', or 'asynchronous'.")

    return spike_trains

def generate_tonic_to_bursting_spike_trains(num_neurons, total_time, dt, tonic_rate, burst_rate, inter_spike_interval, switch_prob, firing_type, avg_spikes_per_burst, sd_spikes_per_burst, tonic_interval_noise, jitter):
    """
    Generate spike trains for hybrid neurons that normally fire tonically but can switch to bursting.

    Args:
        num_neurons (int): Number of neurons.
        total_time (float): Total simulation time in seconds.
        dt (float): Time step in seconds.
        tonic_rate (float): Firing rate for tonic firing (in Hz).
        burst_rate (float): Bursts per second when bursting.
        inter_spike_interval (float): Interval between spikes within a burst (in seconds).
        switch_prob (float): Probability of switching from tonic to burst mode at each time step.
        firing_type (str): Type of firing ("random", "synchronous", "asynchronous").
        avg_spikes_per_burst (int): Average number of spikes per burst.
        sd_spikes_per_burst (int): Standard deviation of spikes per burst.
        tonic_interval_noise (float): Noise as a percentage of the interval for tonic firing.
        jitter (float): Maximum jitter (in seconds) to apply for "synchronous" firing.

    Returns:
        list: List of spike trains for the hybrid neurons.
    """
    time_points = int(total_time / dt)
    mean_interval_between_spikes = int(1 / tonic_rate / dt)  # Mean time points between spikes

    # Generate tonic firing rates and adjust to target mean

    spike_trains = []
    
    base_spike_train = np.zeros(time_points)
    current_time = 0

    if firing_type == "synchronous":
        # Generate base spike train
        base_burst_train = np.zeros(time_points)
        current_time = 0
        tonic_rate = np.random.uniform(0.5, 3, num_neurons)

        # Adjust the firing rates to ensure the mean is exactly 4 Hz
        current_mean = np.mean(tonic_rate)
        target_mean = 2.0
        tonic_rate = tonic_rate + (target_mean - current_mean)

        while current_time < time_points:
            # Decide whether to burst or tonic fire
            if np.random.rand() < switch_prob:
                # Bursting mode
                burst_times = np.random.poisson(burst_rate * dt, 1)[0]
                spikes_in_burst = max(1, int(np.random.normal(avg_spikes_per_burst, sd_spikes_per_burst)))
                for j in range(spikes_in_burst):
                    spike_time = current_time + int(j * inter_spike_interval / dt)
                    if spike_time < time_points:
                        base_burst_train[spike_time] = 1
                current_time += int(spikes_in_burst * inter_spike_interval / dt)
            else:
                # Tonic firing mode
                noisy_interval = mean_interval_between_spikes + int(
                    np.random.normal(0, tonic_interval_noise * mean_interval_between_spikes)
                )
                current_time += max(1, noisy_interval)  # Ensure at least 1 time step between spikes


        max_jitter_points = int(jitter / dt)  # Convert jitter from seconds to time points

        for i in range(num_neurons):
            neuron_spike_train = np.zeros(time_points)
            current_time = 0

            # Tonic firing based on individual neuron firing rate
            mean_interval_between_spikes = int(1 / tonic_rate[i] / dt)

            while current_time < time_points:
                # Tonic firing mode
                noisy_interval = mean_interval_between_spikes + int(
                    np.random.normal(0, 0.1 * mean_interval_between_spikes)
                )
                current_time += max(1, noisy_interval)
                if current_time < time_points:
                    neuron_spike_train[current_time] = 1

            # Combine individual tonic train with shared burst train
            combined_train = np.logical_or(neuron_spike_train, base_burst_train).astype(int)

            # Apply jitter to the combined train
            spike_indices = np.where(combined_train > 0)[0]
            jittered_spike_train = np.zeros(time_points)

            for spike_time in spike_indices:
                jittered_time = spike_time + np.random.randint(-max_jitter_points, max_jitter_points)
                jittered_time = min(max(jittered_time, 0), time_points - 1)
                jittered_spike_train[jittered_time] = 1

            spike_trains.append(jittered_spike_train)

    elif firing_type == "asynchronous":
        spike_trains = []
        for i in range(num_neurons):
            spike_train = np.zeros(time_points)
            current_time = 0

            while current_time < time_points:
                # Decide whether to burst or tonic fire
                if np.random.rand() < switch_prob:
                    # Bursting mode
                    burst_times = np.random.poisson(burst_rate * dt, 1)[0]
                    spikes_in_burst = max(1, int(np.random.normal(avg_spikes_per_burst, sd_spikes_per_burst)))
                    for j in range(spikes_in_burst):
                        spike_time = current_time + int(j * inter_spike_interval / dt)
                        if spike_time < time_points:
                            spike_train[spike_time] = 1
                    current_time += int(spikes_in_burst * inter_spike_interval / dt)
                else:
                    # Tonic firing mode
                    noisy_interval = mean_interval_between_spikes + int(
                        np.random.normal(0, tonic_interval_noise * mean_interval_between_spikes)
                    )
                    current_time += max(1, noisy_interval)  # Ensure at least 1 time step between spikes
                    if current_time < time_points:
                        spike_train[current_time] = 1

            spike_trains.append(spike_train)

    elif firing_type == "random":
        spike_trains = []
        for i in range(num_neurons):
            spike_train = np.random.poisson(tonic_rate * dt, time_points)
            spike_trains.append(spike_train)
    
    else:
        raise ValueError("Invalid firing type. Choose 'random', 'synchronous', or 'asynchronous'.")

    return spike_trains

# Parameters
num_neurons_tonic = 60
num_neurons_bursting = 40
total_time = 60  # Total time in seconds
dt = 0.001  # Time step in seconds (1 ms)
tonic_rate =4  # Firing rate for tonic neurons (3 Hz)
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
#plt.plot(template)
#plt.show()
fs = 1000
nperseg = 4096
noverlap = 3072
max_freq = 30
repeats = 10 
real_data_spectra_path = "data/combined_VS_before_spectra.csv"
real_data_df = pd.read_csv(real_data_spectra_path, skiprows=1, index_col=0)
real_spectra = []
for i in range(real_data_df.shape[0]):
    bulk_calcium_trace = real_data_df.iloc[i].values  # Get the calcium trace for neuron i
    freqs, power_dB = m_a.compute_power_spectrum_dB(bulk_calcium_trace, fs, nperseg=nperseg, noverlap=noverlap, max_freq=max_freq)
    real_spectra.append((freqs, power_dB))



for tonic_firing_type in firing_types:
    for bursting_firing_type in firing_types:
        print(f"Simulating for Tonic: {tonic_firing_type}, Bursting: {bursting_firing_type}")

        all_power_spectra_sim = []  # To store the power spectra for each repetition (simulated)
        all_power_spectra_exp = []  # To store experimental spectra (real data)

        # Repeat the generation 30 times
        for repeat in range(repeats):
            print(f"Repeat {repeat + 1}/{repeats}")

            # Generate tonic spike trains with the current firing type
            tonic_spike_trains = generate_tonic_spike_trains(
                num_neurons=num_neurons_tonic,
                total_time=total_time,
                dt=dt,
                tonic_rate=tonic_rate,
                tonic_interval_noise=tonic_interval_noise,
                firing_type=tonic_firing_type,
                min_isi_time_points=150
            )

            # Simulate bursting neurons with distinct burst patterns
            bursting_spike_trains = generate_tonic_to_bursting_spike_trains(
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

            # Compute the power spectrum for this simulation
            bulk_calcium_trace_ = m_a.robust_zscore(bulk_calcium_trace[1000:])  # Z-score normalization
            freqs_sim, power_dB_sim = m_a.compute_power_spectrum_dB(
                bulk_calcium_trace_, fs, nperseg=nperseg, noverlap=noverlap, max_freq=max_freq
            )

            # Store the power spectrum from this simulation
            all_power_spectra_sim.append(power_dB_sim)
            plot_raster(all_spike_trains,dt)
        # Convert all simulated power spectra into a numpy array for easier manipulation
        all_power_spectra_sim = np.array(all_power_spectra_sim)

        # Compute the mean power spectrum across the 30 repetitions (simulations)
        mean_power_spectrum_sim = np.mean(all_power_spectra_sim, axis=0)

        # Plot all spectra: experimental, simulated, and their means
        plt.figure(figsize=(10, 6))

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
        print(mean_power_spectrum_exp)
        plt.plot(freqs_exp, mean_power_spectrum_exp, color='black', linewidth=4)

        # Add labels and title
        plt.title(f'Spectra for Tonic: {tonic_firing_type}, Bursting: {bursting_firing_type}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.legend()

        # Save the plot
        plt.savefig(f"{output_dir}/combined_spectra_tonic_{tonic_firing_type}_bursting_{bursting_firing_type}.pdf")
        plt.close()

        print(f"Saved combined spectra plot for Tonic: {tonic_firing_type}, Bursting: {bursting_firing_type}")



"""
for tonic_firing_type in firing_types:
    for bursting_firing_type in firing_types:
        print(f"Simulating for Tonic: {tonic_firing_type}, Bursting: {bursting_firing_type}")
        #tonic_spike_trains,bursting_spike_trains=[], []

        # Generate tonic spike trains with the current firing type
        tonic_spike_trains = generate_tonic_spike_trains(
            num_neurons=num_neurons_tonic,
            total_time=total_time,
            dt=dt,
            tonic_rate=tonic_rate,
            tonic_interval_noise=tonic_interval_noise,
            firing_type=tonic_firing_type,
            min_isi_time_points=150
        )

        # Simulate bursting neurons with distinct burst patterns
        bursting_spike_trains = generate_tonic_to_bursting_spike_trains(
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

        # Calculate the bulk spiking rate (sum of spikes across all neurons)
        bulk_spike_train = np.sum(all_spike_trains, axis=0)

        # Plot raster
        plt.figure(figsize=(10, 8))
        plot_raster(all_spike_trains, dt)
        plt.savefig(f"{output_dir}/raster_tonic_{tonic_firing_type}_bursting_{bursting_firing_type}.pdf")
        plt.close()

        # Plot bulk calcium trace
        plt.figure(figsize=(12, 6))
        plt.plot(time_vector, bulk_calcium_trace, label='Bulk ΔF/F')
        plt.xlabel('Time (s)')
        plt.ylabel('ΔF/F (a.u.)')
        plt.title(f'Bulk Calcium Trace for Tonic: {tonic_firing_type}, Bursting: {bursting_firing_type}')
        plt.legend()
        plt.savefig(f"{output_dir}/bulk_calcium_tonic_{tonic_firing_type}_bursting_{bursting_firing_type}.pdf")
        plt.close()

        # Compute and plot power spectrum
        bulk_calcium_trace_ = m_a.robust_zscore(bulk_calcium_trace[1000:])
        freqs_sim, power_dB_sim = m_a.compute_power_spectrum_dB(bulk_calcium_trace_, fs, nperseg=nperseg, noverlap=noverlap, max_freq=max_freq)

        plt.figure(figsize=(10, 4))
        for freqs, power_dB in real_spectra:
            plt.plot(freqs, power_dB, color='grey', alpha=0.5)
        plt.plot(freqs_sim, power_dB_sim, color='blue', label='PS of Sim Release')
        plt.title(f'Power Spectrum for Tonic: {tonic_firing_type}, Bursting: {bursting_firing_type}')
        plt.savefig(f"{output_dir}/power_spectrum_tonic_{tonic_firing_type}_bursting_{bursting_firing_type}.pdf")
        plt.close()

        print(f"Saved plots for Tonic: {tonic_firing_type}, Bursting: {bursting_firing_type}")


"""

"""
ground_truth_df = pd.DataFrame({
    'Time (s)': time_vector[1000:],
    'Bulk ΔF/F': calcium_traces[22][1000:],
    'Bulk Spike Count': all_spike_trains[22][1000:]

})


ground_truth_df['Smoothed Bulk ΔF/F'] = ground_truth_df['Bulk ΔF/F'].rolling(window=50, min_periods=1).mean()
plot_raster(all_spike_trains, dt)
plot_calcium_and_spikes(ground_truth_df, all_spike_trains, dt=dt, neuron_idx=22, start_idx=1000)


plt.figure(figsize=(14, 7))
plt.plot(ground_truth_df['Time (s)'], ground_truth_df['Bulk ΔF/F'], label='Original Signal', alpha=0.6)
plt.plot(ground_truth_df['Time (s)'], ground_truth_df['Smoothed Bulk ΔF/F'], label='Smoothed Signal', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Bulk ΔF/F')
plt.title('Original vs Smoothed Bulk ΔF/F Signal')
plt.legend()
plt.grid(True)
plt.show()
# Save to CSV
ground_truth_df.to_csv('data/ground_truth_data_s.csv', index=False)
"""

