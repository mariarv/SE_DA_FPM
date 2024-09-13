import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import metrics_analysis as m_a

# Parameters
N = 20  # Number of dopaminergic neurons
duration = 20 * b2.second  # Simulation time
dt = 1 * b2.ms  # Simulation time step
print(dt)

tonic_rate = 2* b2.Hz  # Average firing rate for tonic neurons
burst_rate = 10  # Firing rate during burst (dimensionless)
burst_spikes = 3  # Number of spikes in each burst
burst_quiescent_time = 100 * b2.ms  # Quiescent period between bursts

# Define tonic and burst neuron indices
tonic_neurons = int(0.8 * N)
burst_neurons = N - tonic_neurons

# Neuron group
neurons = b2.NeuronGroup(N, '''
dv/dt = (I-v) / tau : 1 
dopamine_release : 1 
I : 1
tau : second
burst_spike_count : integer
burst_active : boolean
last_burst_time : second
''', threshold='v>1', reset='v=0', refractory=5*b2.ms, method='exact')

# Initialization
#neurons.I[:tonic_neurons] = tonic_rate
neurons.burst_active[:tonic_neurons] = False
neurons.I[tonic_neurons:] = 0  # Start with no input
neurons.burst_spike_count[tonic_neurons:] = burst_spikes
neurons.burst_active[tonic_neurons:] = True  # Start burst firing
neurons.last_burst_time[tonic_neurons:] = 0  # Initialize to allow immediate bursting
#neurons.v = 'rand() * 0.5'  # Random initial conditions with lower v for desynchronization
neurons.tau = 50 * b2.ms
# Create individual Poisson inputs for each tonic neuron
poisson_input = b2.PoissonGroup(tonic_neurons, rates=tonic_rate)
tonic_synapses = b2.Synapses(poisson_input, neurons[:tonic_neurons], on_pre='v_post += 0.8')  # Adjust weight as needed
tonic_synapses.connect(j='i')  # Connect each Poisson neuron to one tonic neuron

# Handling dopamine profile
read_da = pd.read_csv("data/Templates/Before_Cocaine_VS_mean_traces_dff.csv")
ORIGINAL_RATE = 1017.252625
TARGET_RATE = 1000
dopamine_profile = m_a.resample_signal(read_da["25"].values, ORIGINAL_RATE, TARGET_RATE)

time_vector = np.arange(0, len(dopamine_profile) * dt, dt)  # Time vector
plt.plot(time_vector / b2.ms, dopamine_profile)  # Plot the resampled dopamine profile

# Spike monitor
spikemon = b2.SpikeMonitor(neurons)
statemon = b2.StateMonitor(neurons, 'v', record=True)
dopamine_mon = b2.StateMonitor(neurons, 'dopamine_release', record=True)

buffer_length = int(duration / dt) + len(dopamine_profile)
dopamine_release_buffer = np.zeros((N, buffer_length))

@b2.network_operation(dt=dt)
def update_dopamine_release():
    global dopamine_release_buffer
    current_time = b2.defaultclock.t

    new_spikes = spikemon.i
    spike_times = spikemon.t / b2.ms

    if len(new_spikes) == 0:
        return

    for idx, neuron_idx in enumerate(new_spikes):
        spike_time = spike_times[idx]
        t_idx = int(spike_time / (dt / b2.ms))

        start_idx = t_idx
        end_idx = min(t_idx + len(dopamine_profile), dopamine_release_buffer.shape[1])
        profile_length = end_idx - start_idx

        if profile_length > 0:
            dopamine_release_buffer[neuron_idx, start_idx:end_idx] += dopamine_profile[:profile_length]

    for i in range(tonic_neurons, N):
        if neurons.burst_active[i]:
            if neurons.burst_spike_count[i] > 0:
                neurons.I[i] = burst_rate
                neurons.burst_spike_count[i] -= 1
            if neurons.burst_spike_count[i] == 0:
                neurons.I[i] = 0  # Stop bursting
                neurons.burst_active[i] = False  # Enter quiescent period
                neurons.last_burst_time[i] += (current_time  -neurons.last_burst_time[i])
        else:
            if current_time - neurons.last_burst_time[i] >= burst_quiescent_time:
                neurons.burst_active[i] = True  # Start next burst after quiescent time
                neurons.burst_spike_count[i] = burst_spikes
                neurons.I[i] = burst_rate  # Restart the burst

    current_idx = int(b2.defaultclock.t / dt)
    if current_idx < buffer_length:
        for neuron_idx in range(N):
            neurons.dopamine_release[neuron_idx] = dopamine_release_buffer[neuron_idx, current_idx]
    else:
        print(f"Warning: current_idx {current_idx} is out of buffer bounds.")

    dopamine_release_buffer *= 0.9

# Create and run the network
net = b2.Network(neurons, spikemon, dopamine_mon, update_dopamine_release, tonic_synapses, poisson_input)
net.run(duration)

# Plot results
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(spikemon.t/b2.ms, spikemon.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Raster plot')

plt.subplot(122)
bulk_dopamine_release = dopamine_mon.dopamine_release.sum(axis=0) / N
plt.plot(dopamine_mon.t/b2.ms, bulk_dopamine_release)
plt.xlabel('Time (ms)')
plt.ylabel('Bulk Dopamine Release (Df/f)')
plt.title('Bulk Dopamine Release')

plt.tight_layout()
plt.show()

fs = 1000
nperseg = 4096
noverlap = 3072
max_freq = 30
freqs, power_dB = m_a.compute_power_spectrum_dB(bulk_dopamine_release, fs, nperseg=nperseg, noverlap=noverlap, max_freq=max_freq)
plt.figure(figsize=(10, 4))
plt.plot(freqs, power_dB, color='blue', label='PS of sim release')
plt.show()
