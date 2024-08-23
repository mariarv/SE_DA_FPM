import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import metrics_analysis as m_a
# Define basic parameters
N = 10  # Number of dopaminergic neurons
duration = 1 * b2.second  # Simulation time
dt = b2.defaultclock.dt  # Simulation time step

# Define the neuron model
eqs = '''
dv/dt = (I-v)/tau : 1
I : 1
tau : second
dopamine_release : 1 
'''

# Create neuron group
G = b2.NeuronGroup(N, model=eqs, threshold='v>1', reset='v=0', method='euler')

# Initialize parameters to generate tonic firing
G.v = 'rand()'  # random initial conditions
G.tau = 50 * b2.ms
G.I = '1.2 + 0.1*sin(2*pi*1*t/second) + 0.05*randn()'  # 1 Hz oscillation with noise

amplitude = 1.0
center =0   # Center of the Gaussian in ms
sigma = 0.01 # Width of the Gaussian in ms
skewness = 0.5  # Skewness factor (unitless)


def simple_gaussian(t, amplitude, center, sigma):
    return amplitude * np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
# Precompute the skewed Gaussian time course for a single spike
time_vector = np.arange(0, 50, dt)  # Time vector from 0 to 50 ms
dopamine_profile = simple_gaussian(time_vector, amplitude, center, sigma)
plt.plot(dopamine_profile)
# Spike monitor to track spikes
spikemon = b2.SpikeMonitor(G)

# State monitor to track dopamine release
dopamine_mon = b2.StateMonitor(G, 'dopamine_release', record=True)
total_simulation_time = 1 * b2.second  # Total simulation time
buffer_length = int(total_simulation_time / dt) + len(dopamine_profile)
dopamine_release_buffer = np.zeros((N, buffer_length))

# Dopamine release mechanism directly driven by spikes
@b2.network_operation(dt=dt)
def update_dopamine_release():
    global dopamine_release_buffer  # Ensure we modify the global buffer
    new_spikes = spikemon.i
    spike_times = spikemon.t / b2.ms

    if len(new_spikes) == 0:
        return


    for idx, neuron_idx in enumerate(new_spikes):
        spike_time = spike_times[idx]
        t_idx = int(spike_time / (dt / b2.ms))  # Ensure correct scaling for t_idx
        
        # Handle buffer overflow gracefully
        start_idx = t_idx
        end_idx = min(t_idx + len(dopamine_profile), dopamine_release_buffer.shape[1])
        
        profile_length = end_idx - start_idx
        if profile_length > 0:
            dopamine_release_buffer[neuron_idx, start_idx:end_idx] += dopamine_profile[:profile_length]
            # Debugging: print buffer updates
           # print(f"Neuron {neuron_idx}: Added dopamine profile at t_idx {t_idx}, start_idx {start_idx}, end_idx {end_idx}")
    # Apply the buffer to the dopamine release at the current time step
    current_idx = int(b2.defaultclock.t / dt)
    if current_idx < buffer_length:
       # print(f"Applying buffer at current_idx {current_idx}")
        for neuron_idx in range(N):
            G.dopamine_release[neuron_idx] = dopamine_release_buffer[neuron_idx, current_idx]
            #print(f"Neuron {neuron_idx} dopamine release at time {b2.defaultclock.t}: {G.dopamine_release[neuron_idx]}")
    else:
        print(f"Warning: current_idx {current_idx} is out of buffer bounds.")

    # Apply decay to the buffer
    dopamine_release_buffer *= 0.9

# Create and run the network
net = b2.Network(G, spikemon, dopamine_mon, update_dopamine_release)
net.run(duration)

# Plot results

# Plot raster plot of neuronal firing
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(spikemon.t/b2.ms, spikemon.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Raster plot')

# Plot the bulk dopamine release (Df/f)
plt.subplot(122)
bulk_dopamine_release = dopamine_mon.dopamine_release.sum(axis=0) / N
plt.plot(dopamine_mon.t/b2.ms, bulk_dopamine_release)
plt.xlabel('Time (ms)')
plt.ylabel('Bulk Dopamine Release (Df/f)')
plt.title('Bulk Dopamine Release')

plt.tight_layout()
plt.show()

fs= 10000
nperseg=4096
noverlap=3072
max_freq=50
freqs, power_dB = m_a.compute_power_spectrum_dB(bulk_dopamine_release, fs, nperseg=nperseg, noverlap=noverlap, max_freq=max_freq)
plt.figure(figsize=(10, 4))
plt.plot(freqs, power_dB, color='blue', label='PS of sim release')
plt.show()

