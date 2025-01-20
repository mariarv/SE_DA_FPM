import os
import numpy as np
import pandas as pd
from tdt import read_block
import matplotlib.pyplot as plt

# Import the new function from your metrics_analysis (adjust path/name if needed)
import metrics_analysis as m_a

# or if you placed FP_normalize_RVZv2 in this script directly, skip above and call it here.

def filter_by_epocs(data, timestamps, tstart, sampling_rate, min_interval=30, padding=3):
    """
    Filter data based on valid time intervals from BOX1 epocs.

    Parameters:
    - data: array-like
        Input data to filter.
    - timestamps: array-like
        Timestamps of BOX1 events.
    - tstart: float
        Start time of the recording.
    - sampling_rate: float
        Sampling rate of the signal (in Hz).
    - min_interval: float
        Minimum time difference (in seconds) between consecutive timestamps to be considered valid.
    - padding: float
        Padding (in seconds) to exclude around each valid interval.

    Returns:
    - concatenated_data: array-like
        Concatenated data for all valid intervals.
    """
    valid_intervals = []
    for i in range(len(timestamps) - 1):
        if timestamps[i + 1] - timestamps[i] >= min_interval:
            valid_intervals.append((timestamps[i] + padding, timestamps[i + 1] - padding))
    
    concatenated_data = []
    for interval in valid_intervals:
        start_idx = max(0, int((interval[0] - tstart) * sampling_rate))
        end_idx = min(len(data), int((interval[1] - tstart) * sampling_rate))
        concatenated_data.extend(data[start_idx:end_idx])
    
    return np.array(concatenated_data)


# Define the base path
base_path = '/Users/reva/Documents/Python/SE_DA_FPM/data/FP dlight for Maria Reva'
output_dir = '/Users/reva/Documents/Python/SE_DA_FPM/output_csv'
os.makedirs(output_dir, exist_ok=True)


# Iterate through tank folders
for tankname in os.listdir(base_path):
    tank_path = os.path.join(base_path, tankname)
    print(tank_path)
    if not os.path.isdir(tank_path) or tankname.startswith('.'):
        continue

    # Iterate through blk subfolders
    for blk_name in os.listdir(tank_path):
        blk_path = os.path.join(tank_path, blk_name)
        if not os.path.isdir(blk_path) or blk_name.startswith('.'):
            continue

        # Find and process .tev files
        tev_files = [f for f in os.listdir(blk_path) if f.endswith('.tev')]
        if not tev_files:
            continue

        # Process the first .tev file
        tev_file = tev_files[0]
        tev_filepath = os.path.join(blk_path, tev_file)

        print(f"Processing file: {tev_filepath}")

        # Read TDT data block
        try:
            data_block = read_block(blk_path)
            # Adjust these stream names based on your actual TDT block
            S = data_block.streams['_470A'].data  # 470 nm signal
            C = data_block.streams['_405A'].data  # 405 nm control
            sampling_rate = data_block.streams['_470A'].fs

            # Just pick a small value for tstart if you don't have a precise one
            tstart = 0.0  
            # Approximate end time:
            tend = len(S) / sampling_rate

            # Pad shorter signal to match lengths if necessary
            if len(S) > len(C):
                C = np.pad(C, (0, len(S) - len(C)), mode='edge')
            elif len(C) > len(S):
                S = np.pad(S, (0, len(C) - len(S)), mode='edge')

            # Build the c_Mag dict (like the MATLAB cont struct)
            # For the baseline, you might choose the entire signal or a subset.
            c_Mag = {
                'data': np.column_stack((S, C)),
                'samplerate': sampling_rate,
                'tstart': tstart,
                'tend': tend,
                'bl_start': 0,          # start index for baseline
                'bl_end': len(S),       # end index for baseline
                'name': blk_name,
                'chanlabels': ['ch470', 'ch405'],
                'nbad_start': 0,
                'nbad_end': 0,
                'max_tserr': 0
            }

            bls = np.polyfit(C, S, 1)
            Y_fit_all = np.multiply(bls[0], C) + bls[1]
            Y_dF_all = S - Y_fit_all
            dFF = np.multiply(100, np.divide(Y_dF_all, Y_fit_all))

            # Example of filtering data by BOX1 epocs
            BOX1 = data_block.epocs['BOX1']  # Replace 'BOX1' with the actual epoc name if different
            timestamps = BOX1.onset         # Onset times

            # Filter data by BOX1 epocs
            concatenated_data_S = filter_by_epocs(S, timestamps, tstart, sampling_rate)
            concatenated_data_C = filter_by_epocs(C, timestamps, tstart, sampling_rate)

            # A quick "manual" approach to compute ΔF/F on the filtered data
            # (This is not the same as the full-blown 'FP_normalize_RVZv2' approach,
            #  but just a demonstration of how to do a quick ratio.)
            concatenated_data = filter_by_epocs(dFF, timestamps, tstart, sampling_rate)

            df = pd.DataFrame({
                'ch405': concatenated_data_C,
                'ch470': concatenated_data_S,
                'df/f': concatenated_data
            })
            csv_filename = os.path.join(output_dir, f"{blk_name}_data_A.csv")
            df.to_csv(csv_filename, index=False)
            print(f"Data saved successfully to {csv_filename}")

            plt.tight_layout()
            #plt.show()

        except Exception as e:
            print(f"Error processing {tev_filepath}: {e}")

tank_data = {}

# Collect data for each tank
for tankname in os.listdir(base_path):
    tank_path = os.path.join(base_path, tankname)
    if not os.path.isdir(tank_path) or tankname.startswith('.'):
        continue

    # Initialize tank data storage
    if tankname not in tank_data:
        tank_data[tankname] = []

    for blk_name in os.listdir(tank_path):
        blk_path = os.path.join(tank_path, blk_name)
        if not os.path.isdir(blk_path) or blk_name.startswith('.'):
            continue

        # Process CSV files saved earlier
        csv_filename = os.path.join(output_dir, f"{blk_name}_data_A.csv")
        if os.path.exists(csv_filename):
            df = pd.read_csv(csv_filename)
            tank_data[tankname].append((blk_name, df))

# Plotting all traces for each tankname
for tankname, traces in tank_data.items():
    num_traces = len(traces)
    plt.figure(figsize=(10, 5 * num_traces))

    for i, (blk_name, df) in enumerate(traces):
        plt.subplot(num_traces, 1, i + 1)
        plt.plot((df['df/f'][2000:]), label=f"{blk_name} DF/F", )
        plt.title(f"Tank: {tankname} - Block: {blk_name}")
        plt.xlabel("Time (samples)")
        plt.ylabel("ΔF/F")
        plt.legend(loc="upper right")

    plt.tight_layout()
    # Save the plot instead of showing it

plt.show()


# Plot power spectra for each tankname
for tankname, traces in tank_data.items():
    for i, (blk_name, df) in enumerate(traces):
        # Prepare the signals
        dfs = df['ch470'][1000:]
        df_s = (dfs - np.mean(dfs))

        dfc = df['ch405'][1000:]
        df_c = (dfc - np.mean(dfc))

        # Compute power spectra
        freqs, power_dB = m_a.compute_power_spectrum_dB(
            df_s, sampling_rate, nperseg=4096, max_freq=30
        )
        freqs, power_dB_c = m_a.compute_power_spectrum_dB(
            df_c, sampling_rate, nperseg=4096, max_freq=30
        )

        # Create a time axis for the traces
        time = np.arange(len(df_s)) / sampling_rate  # Convert sample indices to seconds
        relative_power_s = power_dB / np.sum(power_dB)  # Normalize by total power
        relative_power_c = power_dB_c / np.sum(power_dB_c)

        # Plot both subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Left subplot: Power spectra
        axs[1].plot(freqs, relative_power_s, label=f"470 ", color="r")
        axs[1].plot(freqs, relative_power_c, label=f"405 ", color="k")
        axs[1].set_title("Power Spectra")
        axs[1].set_xlabel("Log(Frequency (Hz))")
        axs[1].set_ylabel("Relatove Power Density")
        axs[1].legend(loc="upper right")
        axs[1].grid(True)

        # Right subplot: Traces as a function of time
        axs[0].plot(time, dfs, label="470 nm", color="r")
        axs[0].plot(time, dfc, label="405 nm", color="k")
        axs[0].set_title("Traces Over Time")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Signal (a.u.)")
        axs[0].legend(loc="upper right")
        axs[0].grid(True)

        # Add a title for the figure including tankname and blk_name
        fig.suptitle(f"Tank: {tankname}, Block: {blk_name}", fontsize=16)

        # Adjust layout and save the figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
        plot_filename = os.path.join(output_dir, f"{tankname}_{blk_name}_spectra_traces A.png")
        plt.savefig(plot_filename)
        print(f"Plot saved at {plot_filename}")
        plt.close()

from matplotlib.pyplot import get_cmap

# Define a color map to assign unique colors for each tankname
color_map = get_cmap('tab10')

# Initialize a new figure for power spectra
plt.figure(figsize=(10, 8))

# Maximum number of colors (adjust if you have more tanknames)
num_colors = 10
tanknames = list(tank_data.keys())
color_dict = {tanknames[i]: color_map(i / num_colors) for i in range(len(tanknames))}

# Plot power spectra for each tankname
for tankname, traces in tank_data.items():
    all_power_dB = []

    for i, (blk_name, df) in enumerate(traces):
        df0=df['df/f'][1000:]
        # Compute power spectrum for this trace
        df_=(df0-np.mean(df0))/np.std(df0)
        freqs, power_dB = m_a.compute_power_spectrum_dB(
            df_, sampling_rate, nperseg=4096, max_freq=30
        )
        all_power_dB.append(power_dB)

    # Average power spectrum for all traces in this tank

    # Plot the averaged spectrum
        plt.plot(freqs, power_dB, label=f"Tank: {blk_name}", color=color_dict[tankname])

# Customize the plot
plt.title("Power Spectra for Each Tank")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.legend(loc="upper right")
plt.grid(True)

# Save the power spectra plot
#power_plot_filename = os.path.join(output_dir, "power_spectra.png")
#plt.savefig(power_plot_filename)
#print(f"Power spectra plot saved at {power_plot_filename}")
plt.show()