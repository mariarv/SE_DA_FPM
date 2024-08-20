import pandas as pd
import numpy as np
from brian2 import *
from scipy.signal import welch
import matplotlib.pyplot as plt
import pickle
import os
# Load the release profile from the pickle file
TEMPLATE_FILE_PATH = 'data/Templates/After_Cocaine_DS_mean_traces_dff.csv'
def load_template(template_file_path, dur):
    if not os.path.exists(template_file_path):
        logging.error(f"File not found: {template_file_path}")
        raise FileNotFoundError(f"File not found: {template_file_path}")
    template_ = pd.read_csv(template_file_path)
    template = template_[dur].values
    logging.info(f"Loaded template from {template_file_path}")
    return np.array(template)

# Assuming df_combined_ds contains the release profile as a NumPy array
release_profile = load_template(TEMPLATE_FILE_PATH, "25")

# Define parameters
N = 10  # Single DA neuron
duration = 10000*ms  # Duration of simulation
dt = 0.1*ms  # Time step

# Define the neuron model
eqs = '''
dv/dt = (I-v) / tau : 1 (unless refractory)
I : 1
tau : second
'''

# Create the neuron group
neuron = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', refractory=5*ms, method='exact')

# Set the parameters for tonic and phasic firing
neuron.I = 10  # Adjusted to match desired firing rate
neuron.tau = 10*ms

# Record spikes and dopamine release
spikes = SpikeMonitor(neuron)
trace = StateMonitor(neuron, 'v', record=True)

# Define dopamine release model based on photometry data
def dopamine_release(spike_times, dt, release_profile):
    release_profile=release_profile/ np.max(release_profile)
    release = np.zeros(int(duration/dt))
    for spike in spike_times:
        start_idx = int(spike/dt)
        end_idx = start_idx + len(release_profile)
        if end_idx < len(release):
            release[start_idx:end_idx] += release_profile
    return release

# Run the simulation
run(duration)

# Get spike times
spike_times = spikes.t/ms

# Compute dopamine release
dopamine_release_trace = dopamine_release(spike_times, dt, release_profile)

# Perform spectral analysis
f, Pxx = welch(dopamine_release_trace, fs=1000/dt)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.plot(dopamine_release_trace)
plt.title('Dopamine Release Trace')
plt.xlabel('Time (ms)')
plt.ylabel('Dopamine Release')

plt.subplot(122)
plt.plot(f, Pxx)
plt.title('Power Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')

plt.show()
