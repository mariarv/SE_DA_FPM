#analysis of the Groves data
# Plot 
#   - ordered heatmap 
#   - average trace
#   - spectra of avg trace and dynamics


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import itertools
import metrics_analysis as m_a
from scipy.ndimage import gaussian_filter1d
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.patches as patches
from scipy.stats import binom
from matplotlib_venn import venn2
from upsetplot import from_contents, UpSet


fs = 4
nperseg = 32
noverlap = 64
max_freq = 30

IG_data = pd.read_csv("/Users/reva/Documents/Python/SE_DA_FPM/data/Grove data MIniScope/waterIG.csv", header=0)
def zscore_rows(matrix):
    return (matrix - np.nanmean(matrix, axis=1, keepdims=True)) / np.nanstd(matrix, axis=1, keepdims=True)

time_stamps = IG_data.iloc[0, 1:601].values  # First row (except first column) are time stamps, limited to first 600 columns
IG_data = IG_data.iloc[1:, :]  # Remove the first row (time stamps)
IG_data.columns = ['ID'] + list(IG_data.columns[1:])  # Rename columns (ID + time stamps)

# Step 3: Group Data by Subject ID
grouped_data = IG_data.groupby('ID')


# Function to calculate correlation matrix
def calculate_correlation(matrix):
    return np.corrcoef(matrix)

# Function to find top synchronized neurons
def find_best_synchronized_trio(correlation_matrix):
    num_neurons = correlation_matrix.shape[0]
    max_avg_corr = -np.inf
    best_trio = None

    # Check all combinations of three neurons
    for trio in itertools.combinations(range(num_neurons), 3):
        avg_corr = (
            correlation_matrix[trio[0], trio[1]]
            + correlation_matrix[trio[0], trio[2]]
            + correlation_matrix[trio[1], trio[2]]
        ) / 3
        if avg_corr > max_avg_corr:
            max_avg_corr = avg_corr
            best_trio = trio

    return best_trio
def compute_synchrony_index(matrix, window_size):
    """
    Compute the synchrony index (mean pairwise correlation) over sliding windows.
    
    Args:
    - matrix: 2D array (neurons × time) of neural activity.
    - window_size: Size of the sliding window in time points.
    
    Returns:
    - synchrony_index: List of synchrony values for each window.
    """
    num_neurons, num_timepoints = matrix.shape
    synchrony_index = []
    
    # Loop through time windows
    for t in range(0, num_timepoints - window_size, window_size):
        # Extract the window of activity
        window = matrix[:, t:t + window_size]
        
        # Compute correlation matrix for the current window
        correlation_matrix = np.corrcoef(window)
        
        # Exclude the diagonal (self-correlations) and calculate mean pairwise correlation
        avg_corr = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        synchrony_index.append(avg_corr)
    
    return synchrony_index


# Function to plot interactive 3D trajectory
def plot_3d_trajectory(neurons, title, time_points):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory
    ax.plot(neurons[0], neurons[1], neurons[2], color='blue', alpha=0.7, linewidth=1)
    scatter = ax.scatter(neurons[0], neurons[1], neurons[2], c=time_points, cmap="viridis", s=10)
    
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel("Neuron 1 Activity")
    ax.set_ylabel("Neuron 2 Activity")
    ax.set_zlabel("Neuron 3 Activity")
    fig.colorbar(scatter, label="Time")
    plt.show()

# Example: Analyze and plot 3D trajectories for each subject
for subject_id, subject_data in grouped_data:
    # Extract the activity matrix (neurons × time)
    matrix = subject_data.iloc[:, 1:601].values.astype(float)
    time_points = np.arange(matrix.shape[1])  # Time points for coloring
    
    # Step 1: Compute correlation matrix
    correlation_matrix = calculate_correlation(matrix)
    
    best_trio = find_best_synchronized_trio(correlation_matrix)
    best_trio_activity = matrix[list(best_trio)]
    
    # Step 3: Pick 3 random neurons
    random_neurons = np.random.choice(matrix.shape[0], 3, replace=False)
    random_neurons_activity = matrix[random_neurons]
   # Step 4: Plot 3D trajectory for the most synchronized trio
    print(f"Subject {subject_id}: Best Synchronized Trio {best_trio}")
    #plot_3d_trajectory(best_trio_activity, f"3D Trajectory (Best Synchronized Trio) - Subject {subject_id}", time_points)
    
    # Step 5: Plot 3D trajectory for random neurons
    print(f"Subject {subject_id}: Random Neurons {random_neurons}")
    #plot_3d_trajectory(random_neurons_activity, f"3D Trajectory (Random Neurons) - Subject {subject_id}", time_points)



# Function to order neurons based on synchrony
def order_neurons_by_synchrony(correlation_matrix):
    avg_correlation = np.mean(correlation_matrix, axis=1)  # Average correlation for each neuron
    sorted_indices = np.argsort(avg_correlation)[::-1]  # Sort neurons by decreasing synchrony
    return sorted_indices

# Function to plot heatmap of reordered data
def plot_data_heatmap(matrix, title="Heatmap of Neural Activity (Ordered by Synchrony)"):
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, cmap='bwr', center=0, cbar_kws={'label': 'Z-score'})
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Neurons (Ordered by Synchrony)")
    plt.show()


def plot_average_activity(data, title="Average Activity of Subpopulation"):
    avg_activity = np.mean(data, axis=0)
    plt.figure(figsize=(12, 6))
    plt.plot(avg_activity, color='blue', label='Average Activity')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Activity (Average)")
    plt.legend()
    plt.show()


def plot_synchrony_index(data, window_size=50, title="Synchrony Index Over Time"):
    num_neurons, num_timepoints = data.shape
    synchrony_index = []
    for t in range(0, num_timepoints - window_size, window_size):
        window = data[:, t:t+window_size]
        correlation_matrix = np.corrcoef(window)
        avg_corr = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        synchrony_index.append(avg_corr)
    
    plt.figure(figsize=(12, 6))
    plt.plot(synchrony_index, color='red', label='Synchrony Index')
    plt.title(title)
    plt.xlabel("Time Window")
    plt.ylabel("Synchrony Index (Mean Pairwise Correlation)")
    plt.legend()
    #plt.show()

def plot_heatmap_with_average(matrix, title="Heatmap with Average Activity Overlay",name="Heatmap with Average Activity Overlay"):
    """
    Plots a heatmap of the data matrix with an overlaid average activity trace.
    
    Args:
    - matrix: 2D array-like (neurons × time) representing neural activity.
    - title: Title of the plot.
    """
    # Step 1: Compute average activity
    avg_activity = np.mean(matrix, axis=0)  # Average across neurons
    correlation_matrix = calculate_correlation(matrix)
    
    # Step 2: Order neurons by synchrony
    sorted_indices = order_neurons_by_synchrony(correlation_matrix)
    ordered_matrix = matrix[sorted_indices, :]  # Reorder the rows (neurons)
    # Step 2: Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 10),
        gridspec_kw={'height_ratios': [1, 4]},
        constrained_layout=True  # Ensure tight layout
    )

    # Step 3: Plot the average activity trace
    ax1.plot(avg_activity, color='black', linewidth=2)
    ax1.set_ylabel("Average Z-score")
    ax1.set_title("Average Activity")
    ax1.set_xlim(0, matrix.shape[1])  # Match the x-axis limits with the heatmap
    ax1.grid(True)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-ticks

    # Step 4: Plot the heatmap
    sns.heatmap(ordered_matrix, cmap='bwr', center=0, cbar_kws={'label': 'Z-score'}, ax=ax2, rasterized=True)
    ax2.set_title("Neural Activity Heatmap")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Neurons")
    ax2.set_xlim(0, matrix.shape[1])  # Explicitly match the x-axis limits

    # Step 5: Set the overall title
    fig.suptitle(title, fontsize=16)
    plt.savefig(name)

def identify_synchronized_neurons(correlation_matrix, threshold=0.5):
    """
    Identify the subpopulation of synchronized neurons based on a correlation threshold.
    
    Args:
    - correlation_matrix: 2D array of pairwise correlations between neurons.
    - threshold: Minimum average correlation to include a neuron in the subpopulation.
    
    Returns:
    - subpopulation_indices: List of indices for synchronized neurons.
    """
    avg_correlation = np.mean(correlation_matrix, axis=1)
    subpopulation_indices = np.where(avg_correlation > threshold)[0]
    return subpopulation_indices

def compute_synchrony_index_subpopulation(matrix, subpopulation_indices, window_size=50):
    """
    Compute the synchrony index for a subpopulation of neurons.
    
    Args:
    - matrix: 2D array (neurons × time) of neural activity.
    - subpopulation_indices: Indices of neurons in the synchronized subpopulation.
    - window_size: Size of the sliding window in time points.
    
    Returns:
    - synchrony_index: List of synchrony values for the subpopulation over time.
    """
    subpopulation_matrix = matrix[subpopulation_indices, :]
    return compute_synchrony_index(subpopulation_matrix, window_size)

def plot_combined_figure(matrix, avg_activity, synchrony_index, freqs, power_dB, title="Combined Analysis"):
    """
    Create a single figure combining:
    - Average activity
    - Neural activity heatmap
    - Synchrony index
    - Power spectrum
    
    Args:
    - matrix: 2D array (neurons × time) for heatmap
    - avg_activity: 1D array for average activity
    - synchrony_index: 1D array for synchrony index over time
    - freqs: 1D array for frequency values (power spectrum)
    - power_dB: 1D array for power in dB (power spectrum)
    - title: Overall title of the figure
    """
    # Create the figure and subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], wspace=0.3, hspace=0.3)

    # Top-left: Average Activity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(avg_activity, color='black', linewidth=2)
    ax1.set_title("Average Activity")
    ax1.set_ylabel("Average Z-score")
    ax1.grid(True)

    # Bottom-left: Heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    sns.heatmap(matrix, cmap='bwr', center=0, cbar_kws={'label': 'Z-score'}, ax=ax2, rasterized=True)
    ax2.set_title("Neural Activity Heatmap")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Neurons")

    # Top-right: Synchrony Index
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.plot(synchrony_index, color='red', linewidth=2)
    ax3.set_title("Synchrony Index")
    ax3.set_xlabel("Time Window")
    ax3.set_ylabel("Mean Correlation")
    ax3.grid(True)

    # Bottom-right: Power Spectrum
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(freqs[1:], power_dB[1:], color='blue', linewidth=2)
    ax4.set_title("Power Spectrum")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Power (dB)")
    ax4.grid(True)

    # Set the overall title
    fig.suptitle(title, fontsize=16, y=0.95)

    # Show the plot
    plt.tight_layout()
    plt.show()

def count_synchronized_neurons(matrix, threshold=0.5):
    """
    Count the number of synchronized neurons based on a correlation threshold.
    
    Args:
    - matrix: 2D array (neurons × time) of neural activity.
    - threshold: Correlation threshold to define synchronization.
    
    Returns:
    - num_synchronized: Number of neurons exceeding the threshold.
    - avg_correlation_per_neuron: Array of average correlations for each neuron.
    """
    # Compute the correlation matrix
    correlation_matrix = np.corrcoef(matrix)
    
    # Calculate the maximum correlation for each neuron (excluding self-correlation)
    np.fill_diagonal(correlation_matrix, -np.inf)  # Ignore diagonal (self-correlation)
    max_correlation_per_neuron = np.max(correlation_matrix, axis=1)
    
    # Count neurons with maximum correlation above the threshold
    num_synchronized = np.sum(max_correlation_per_neuron > threshold)
    
    return num_synchronized


power_spectra_data = []

# Define color mapping for the number of neurons
cmap = plt.get_cmap("viridis")


# Example: Analyze and plot ordered heatmaps for each subject
for subject_id, subject_data in grouped_data:
    # Extract the activity matrix (neurons × time)
    matrix = subject_data.iloc[:, 1:601].values.astype(float)
    num_neurons = matrix.shape[0]

    # Step 1: Compute correlation matrix
    correlation_matrix = calculate_correlation(matrix)
    
    # Step 2: Order neurons by synchrony
    sorted_indices = order_neurons_by_synchrony(correlation_matrix)
    ordered_matrix = matrix[sorted_indices, :]  # Reorder the rows (neurons)
    
    # Step 3: Plot the heatmap of reordered data
    print(f"Subject {subject_id}: Neural activity heatmap ordered by synchrony")
    #plot_data_heatmap(ordered_matrix, title=f"Neural Activity Heatmap (Ordered) - Subject {subject_id}")
    avg_activity = np.mean(matrix, axis=0)  # Average across neurons
    avg_activity=(avg_activity-np.mean(avg_activity))/np.std(avg_activity)

    subpopulation_indices = identify_synchronized_neurons(correlation_matrix, threshold=0.1)

    synchrony_index = compute_synchrony_index_subpopulation(matrix, subpopulation_indices, window_size=2)
    """
    # Step 4: Plot synchrony index for subpopulation
    plt.figure(figsize=(12, 6))
    plt.plot(synchrony_index, color='green', linewidth=2)
    plt.title(f"Synchrony Index Over Time (Subpopulation) - Subject {subject_id}")
    plt.xlabel("Time Window")
    plt.ylabel("Synchrony Index (Mean Pairwise Correlation)")
    plt.grid(True)
    plt.show()
    """
    #plot_heatmap_with_average(subject_data.iloc[:, 1:601].values, title=f"Average Activity of Subpopulation {subject_id}", name=f"/Users/reva/Documents/Python/SE_DA_FPM/results/Groves_miniscope_VTA/waterIG_mouse_{subject_id}.pdf")
    freqs, power_dB = m_a.compute_power_spectrum_dB(avg_activity, fs, nperseg=nperseg, max_freq=max_freq)

    power_spectra_data.append([subject_id, num_neurons] + list(power_dB))

    #plt.figure(figsize=(12, 6))

    #plt.plot(freqs[1:],power_dB[1:] )
    #plt.xlabel("Log(Freq (HZ))")
    #plt.ylabel("Power Spectra")
    #plt.show()
    #avg_activity = gaussian_filter1d(avg_activity, sigma=2)  # Apply Gaussian smoothing
    """
    H_before = m_a.create_sliding_windows(avg_activity, 6, 1)
    H_before_flat = H_before.reshape(-1, H_before.shape[1])
    U, S, Vt = np.linalg.svd(H_before_flat, full_matrices=False)
    num_modes = 3  # Number of modes to retain
    umap_before = U
    time=np.arange(len(avg_activity)) / fs

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])  # 2 rows, 1 column; adjust ratios as needed

    # First subplot: 3D plot
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax1.plot(umap_before[:, 0], umap_before[:, 1], umap_before[:, 2], marker='o', markersize=1, linestyle='-', color='blue')
    ax1.set_title("3D PTC Plot")
    ax1.grid(False)
    ax1.set_axis_off()
    #ax1.set_xlabel('PTC Dimension 1')
    #ax1.set_ylabel('UMAP Dimension 2')
    #ax1.set_zlabel('UMAP Dimension 3')

    # Second subplot: Calcium trace
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time,avg_activity, color='orange')
    ax2.set_title("Bulk DA Trace")
    ax2.set_xlabel('Time(s)')
    ax2.set_ylabel('Normalized')
    plt.show()
    """

    num_synchronized = count_synchronized_neurons(matrix, threshold=0.5)
    
    # Print results
    print(f"Subject {subject_id}:")
    print(f"  Total Neurons: {matrix.shape[0]}")
    print(f"  Synchronized Neurons (r > 0.5): {num_synchronized}")
    print(f"  Percentage of Synchronized Neurons: {num_synchronized / matrix.shape[0] * 100:.2f}%")

columns = ['Subject ID', 'Number of Neurons'] + [f'Log_Power_{round(freq, 2)}' for freq in freqs]
power_spectra_df = pd.DataFrame(power_spectra_data, columns=columns)
output_csv_path = '/Users/reva/Documents/Python/SE_DA_FPM/results/Groves_miniscope_VTA/power_spectra_log.csv'  # Change this to your desired file path
power_spectra_df.to_csv(output_csv_path, index=False)
print(f"Power spectra saved to {output_csv_path}")

# Step 4: Plot all power spectra
plt.figure(figsize=(12, 8))

# Normalize the number of neurons for color coding
num_neurons_list = power_spectra_df['Number of Neurons']
norm = Normalize(vmin=num_neurons_list.min(), vmax=num_neurons_list.max())
sm = ScalarMappable(cmap=cmap, norm=norm)

# Plot each spectrum
for idx, row in power_spectra_df.iterrows():
    power_dB = row[2:]  # Skip 'Subject ID' and 'Number of Neurons'
    num_neurons = row['Number of Neurons']
    plt.plot(freqs, power_dB, color=cmap(norm(num_neurons)), alpha=0.7, label=f"ID: {row['Subject ID']}")

# Add color bar
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label("Number of Neurons")

# Finalize plot
plt.xlabel("Log(Frequency (Hz))")
plt.ylabel("Power Spectra (dB")
plt.title("Log-Scaled Power Spectra Across Subjects (Color-Coded by Number of Neurons)")
plt.grid(True)
plt.tight_layout()
plt.show()




def plot_pca_results(matrix, n_components=2, cluster_labels=None):
    """
    Perform PCA and plot the results in 2D or 3D.
    
    Args:
    - matrix: 2D array (neurons × time) of neural activity.
    - n_components: Number of PCA components (2 for 2D, 3 for 3D).
    - cluster_labels: Optional cluster labels to color the points.
    """
    # Step 1: Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(matrix)
    
    # Step 2: Plot the PCA results
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            pca_result[:, 0], 
            pca_result[:, 1], 
            c=cluster_labels, 
            cmap='viridis', 
            s=50, 
            alpha=0.8
        )
        plt.colorbar(scatter, label='Cluster Labels')
        plt.title("PCA Results (2D)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True)
        plt.show()
    elif n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            pca_result[:, 0], 
            pca_result[:, 1], 
            pca_result[:, 2], 
            c=cluster_labels, 
            cmap='viridis', 
            s=50, 
            alpha=0.8
        )
        fig.colorbar(scatter, label='Cluster Labels')
        ax.set_title("PCA Results (3D)")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")
        plt.show()

###################################################   Detecting synch events  ################################################################

def detect_synchronous_events(matrix, z_threshold_pos=2.0,z_threshold_neg=2.0, min_neurons=3, window_size=2):
    """
    Identify groups of neurons participating in synchronous events.
    
    Args:
    - matrix: 2D array (neurons × time) of z-scored neural activity.
    - z_threshold: Z-score threshold to define high/low activity.
    - min_neurons: Minimum number of neurons required to define an event.
    - window_size: Number of neighboring time points to group into one event.
    
    Returns:
    - high_activity_events: List of time indices for high-activity events.
    - low_activity_events: List of time indices for low-activity events.
    - high_activity_neurons: List of neuron groups active in high-activity events.
    - low_activity_neurons: List of neuron groups active in low-activity events.
    """
    num_neurons, num_timepoints = matrix.shape

    # Step 1: Detect time points with high/low activity
    high_activity = np.where(matrix > z_threshold_pos, 1, 0)
    low_activity = np.where(matrix < z_threshold_neg, 1, 0)

    # Step 2: Identify synchronous time points
    high_activity_times = np.where(np.sum(high_activity, axis=0) >= min_neurons)[0]
    low_activity_times = np.where(np.sum(low_activity, axis=0) >= min_neurons)[0]

    # Step 3: Group time points into events (using sliding window)
    def group_events(times, window_size):
        events = []
        event = [times[0]]
        for t in times[1:]:
            if t - event[-1] <= window_size:
                event.append(t)
            else:
                # Keep only if event spans at least the window size
                if len(event) >= 2:
                    events.append(event)
                event = [t]
        # Add the last event if it meets the size requirement
        if len(event) >= window_size:
            events.append(event)
        return events

    high_activity_events = group_events(high_activity_times, window_size)
    low_activity_events = group_events(low_activity_times, window_size)

    # Step 4: Identify neurons active during each event
    def extract_neurons(events, activity_matrix):
        event_neurons = []
        for event in events:
            active_neurons = np.where(np.sum(activity_matrix[:, event], axis=1) > 0)[0]
            event_neurons.append(active_neurons)
        return event_neurons

    high_activity_neurons = extract_neurons(high_activity_events, high_activity)
    low_activity_neurons = extract_neurons(low_activity_events, low_activity)

    return high_activity_events, low_activity_events, high_activity_neurons, low_activity_neurons

def plot_heatmap_with_events(matrix, high_events, low_events, title="Heatmap with Events"):
    """
    Plot a heatmap of neural activity with synchronous events overlaid.
    
    Args:
    - matrix: 2D array (neurons × time) of neural activity.
    - high_events: List of lists of time indices for high-activity events.
    - low_events: List of lists of time indices for low-activity events.
    - title: Title of the plot.
    """
    avg_activity = np.mean(matrix, axis=0)  # Average across neurons

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 10),
        gridspec_kw={'height_ratios': [1, 4]},
        constrained_layout=True  # Ensure tight layout
    )

    # Step 3: Plot the average activity trace
    ax1.plot(avg_activity, color='black', linewidth=2)
    ax1.set_ylabel("Average Z-score")
    ax1.set_title("Average Activity")
    ax1.set_xlim(0, matrix.shape[1])  # Match the x-axis limits with the heatmap
    ax1.grid(True)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-ticks

    # Step 4: Plot the heatmap
    sns.heatmap(ordered_matrix, cmap='bwr', center=0, cbar_kws={'label': 'Z-score'}, ax=ax2, rasterized=True)
    ax2.set_title("Neural Activity Heatmap")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Neurons")
    ax2.set_xlim(0, matrix.shape[1])  # Explicitly match the x-axis limits

    # Step 5: Set the overall title


    # Overlay high-activity events
    for event in high_events:
        event_start = event[0]
        event_end = event[-1]
        rect = patches.Rectangle(
            (event_start, 0),  # Bottom-left corner
            event_end - event_start + 1,  # Width (time points)
            matrix.shape[0],  # Height (all neurons)
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
            label="High-Activity Event" if "High-Activity Event" not in ax2.get_legend_handles_labels()[1] else None
        )
        ax2.add_patch(rect)
    
    # Overlay low-activity events
    for event in low_events:
        event_start = event[0]
        event_end = event[-1]
        rect = patches.Rectangle(
            (event_start, 0),  # Bottom-left corner
            event_end - event_start + 1,  # Width (time points)
            matrix.shape[0],  # Height (all neurons)
            linewidth=1.5,
            edgecolor="blue",
            facecolor="none",
            linestyle="-.",
            label="Low-Activity Event" if "Low-Activity Event" not in ax2.get_legend_handles_labels()[1] else None
        )
        ax2.add_patch(rect)

    # Add legend for events
    handles, labels = ax2.get_legend_handles_labels()
    if labels:
        ax2.legend(handles, labels, loc="upper right", fontsize=12)
    
    plt.show()

def calculate_max_firing_neurons(N, p, confidence=0.99):
    """
    Calculate the maximum number of neurons that can fire simultaneously.
    
    Args:
    - N: Total number of neurons.
    - p: Probability of a single neuron firing at a specific time point.
    - confidence: Confidence level (e.g., 0.99 for 99%).
    
    Returns:
    - k_max: Maximum number of neurons that can fire simultaneously.
    """
    # Calculate the 99th percentile of the binomial distribution
    k_max = binom.ppf(confidence, N, p)
    return int(np.ceil(k_max)) 

def estimate_firing_probability(matrix, threshold=2.0):
    """
    Estimate the firing probability of a single neuron based on Z-score threshold.
    
    Args:
    - matrix: 2D array (neurons × time) of Z-scored neural activity.
    - threshold: Z-score threshold to define a firing event.
    
    Returns:
    - firing_probability: Average probability of a single neuron firing.
    """
    # Count time points where activity exceeds the threshold for each neuron
    num_neurons, num_timepoints = matrix.shape
    firing_counts = np.sum(matrix > threshold, axis=1)
    
    # Compute the firing probability for each neuron
    neuron_probabilities = firing_counts / num_timepoints
    
    # Average probability across all neurons
    firing_probability = np.mean(neuron_probabilities)
    return firing_probability

def analyze_event_dynamics(matrix, events, z_threshold_pos=2.0, z_threshold_neg=-2.0):
    """
    Analyze the dynamics of positively and negatively synchronized neurons during events.
    
    Args:
    - matrix: 2D array (neurons × time) of Z-scored neural activity.
    - events: List of event time indices (each event is a list of time points).
    - z_threshold_pos: Z-score threshold for positive synchronization.
    - z_threshold_neg: Z-score threshold for negative synchronization.
    
    Returns:
    - event_analysis: List of dictionaries, each containing:
        - 'event_times': Time points in the event.
        - 'pos_neurons_count': Number of positively synchronized neurons (>z_threshold_pos).
        - 'neg_neurons_count': Number of negatively synchronized neurons (<z_threshold_neg).
        - 'pos_time_course': Average Z-score of positively synchronized neurons over time.
        - 'neg_time_course': Average Z-score of negatively synchronized neurons over time.
    """
    event_analysis = []
    for event in events:
        event_matrix = matrix[:, event]  # Extract activity within the event window
        
        # Identify positively and negatively synchronized neurons
        pos_neurons = np.where(np.max(event_matrix, axis=1) > z_threshold_pos)[0]
        neg_neurons = np.where(np.min(event_matrix, axis=1) < z_threshold_neg)[0]
        
        # Count synchronized neurons
        pos_neurons_count = len(pos_neurons)
        neg_neurons_count = len(neg_neurons)
        
        # Calculate time course of activation/deactivation
        pos_time_course = np.mean(event_matrix[pos_neurons], axis=0) if len(pos_neurons) > 0 else np.zeros(len(event))
        neg_time_course = np.mean(event_matrix[neg_neurons], axis=0) if len(neg_neurons) > 0 else np.zeros(len(event))
        
        # Save analysis for this event
        event_analysis.append({
            'event_times': event,
            'pos_neurons_count': pos_neurons_count,
            'neg_neurons_count': neg_neurons_count,
            'pos_time_course': pos_time_course,
            'neg_time_course': neg_time_course
        })
    
    return event_analysis

########################################################### Neuron ID for the sanc events ##################################################################
def plot_upset_diagram(event_analysis, title="Neuron Overlap Across Events"):
    """
    Plot an UpSet diagram to show neuron overlap across multiple events.
    
    Args:
    - event_analysis: List of dictionaries, each containing neuron indices for events.
    - title: Title of the plot.
    """
    # Create a dictionary with event indices and their respective neurons
    event_dict = {
        f"Event {i+1}": set(np.where(np.max(matrix[:, event['event_times']], axis=1) > 2)[0])
        for i, event in enumerate(event_analysis)
    }
    
    # Convert to UpSet plot input format
    upset_data = from_contents(event_dict)
    
    # Create and plot the UpSet diagram
    plt.figure(figsize=(12, 6))
    UpSet(upset_data, subset_size='count', show_counts=True).plot()
    plt.suptitle(title, fontsize=16)
    plt.show()



all_pos_counts = []
all_neg_counts = []
# Example analysis for a subject
for subject_id, subject_data in grouped_data:
    # Extract the activity matrix (neurons × time)
    matrix = subject_data.iloc[:, 1:601].values.astype(float)
    matrix = (matrix - np.mean(matrix, axis=1, keepdims=True)) / np.std(matrix, axis=1, keepdims=True)  # Z-score
    correlation_matrix = calculate_correlation(matrix)
    # Estimate firing probability
    p = estimate_firing_probability(matrix, threshold=2.0)
    N = matrix.shape[0]  # Total number of neurons
    k_max = calculate_max_firing_neurons(N, p, confidence=0.99)
    
    print(f"Subject {subject_id}:")
    print(f"  Estimated Firing Probability (p): {p:.6f}")
    print(f"  Maximum Number of Neurons Firing Simultaneously (99% Confidence): {k_max}")
    # Step 1: Detect synchronous events
    z_threshold_pos = 2
    z_threshold_neg = -2
    num_neurons = matrix.shape[0]
    min_neurons =k_max
    window_size = 5
    high_events, low_events, high_neurons, low_neurons = detect_synchronous_events(
        matrix, z_threshold_pos=z_threshold_pos,z_threshold_neg=z_threshold_neg, min_neurons=min_neurons, window_size=window_size
    )
    
    print(f"Subject {subject_id}:")
    print(f"  High-Activity Events: {len(high_events)}")
    print(f"  Low-Activity Events: {len(low_events)}")
    # Step 2: Order neurons by synchrony
    sorted_indices = order_neurons_by_synchrony(correlation_matrix)
    ordered_matrix = matrix[sorted_indices, :]  # Reorder the rows (neurons)
    plot_heatmap_with_events(
        ordered_matrix,
        high_events=high_events,
        low_events=low_events,
        title=f"Neural Activity Heatmap with Events - Subject {subject_id}"
    )

    high_event_analysis = analyze_event_dynamics(matrix, high_events, z_threshold_pos, z_threshold_neg)
    
    # Analyze low-activity events
    low_event_analysis = analyze_event_dynamics(matrix, low_events, z_threshold_pos, z_threshold_neg)
    
    # Print and visualize results
    print(f"Subject {subject_id}:")
    print(f"  High-Activity Events: {len(high_event_analysis)}")
    print(f"  Low-Activity Events: {len(low_event_analysis)}")
    
    for i, event in enumerate(high_event_analysis[:2]):  # Visualize first 5 high-activity events
        plt.figure(figsize=(12, 6))
        plt.plot(event['event_times'], event['pos_time_course'], label='Positive Synchronization', color='red')
        plt.plot(event['event_times'], event['neg_time_course'], label='Negative Synchronization', color='blue')
        plt.title(f"High-Activity Event {i+1} - Subject {subject_id}")
        plt.xlabel("Time Points")
        plt.ylabel("Average Z-Score")
        plt.legend()
        plt.grid(True)
        plt.show()

    for i, event in enumerate(low_event_analysis[:2]):  # Visualize first 5 low-activity events
        plt.figure(figsize=(12, 6))
        plt.plot(event['event_times'], event['pos_time_course'], label='Positive Synchronization', color='red')
        plt.plot(event['event_times'], event['neg_time_course'], label='Negative Synchronization', color='blue')
        plt.title(f"Low-Activity Event {i+1} - Subject {subject_id}")
        plt.xlabel("Time Points")
        plt.ylabel("Average Z-Score")
        plt.legend()
        plt.grid(True)
        plt.show()

    all_pos_counts.extend([event['pos_neurons_count'] for event in high_event_analysis])
    all_neg_counts.extend([event['neg_neurons_count'] for event in low_event_analysis])

    pos_neurons = set()
    for event in high_event_analysis:
        pos_neurons.update(np.where(np.max(matrix[:, event['event_times']], axis=1) > z_threshold_pos)[0])

    neg_neurons = set()
    for event in low_event_analysis:
        neg_neurons.update(np.where(np.min(matrix[:, event['event_times']], axis=1) < z_threshold_neg)[0])

    # Plot Venn diagrams for positive events
    print(f"Subject {subject_id}: Positive Events Venn Diagrams")
    plot_upset_diagram(high_event_analysis, title="Neuron Overlap Across Positive Synchronization Events")

    # Plot Venn diagrams for negative events
    print(f"Subject {subject_id}: Negative Events Venn Diagrams")
    plot_upset_diagram(low_event_analysis, title="Neuron Overlap Across Negative Synchronization Events")

all_pos_counts = np.array(all_pos_counts)
all_neg_counts = np.array(all_neg_counts)

# Plot distributions
plt.figure(figsize=(12, 6))
plt.hist(all_pos_counts, bins=20, alpha=0.7, label='Positive Neurons Count (>2Z)', color='red', density=True)
plt.hist(all_neg_counts, bins=20, alpha=0.7, label='Negative Neurons Count (<-2Z)', color='blue', density=True)
plt.title("Distribution of Synchronized Neurons Across All Subjects")
plt.xlabel("Number of Synchronized Neurons")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(12, 6))
sns.kdeplot(all_pos_counts, color='red', label='Positive Neurons Count (>2Z)', fill=True, alpha=0.4)
sns.kdeplot(all_neg_counts, color='blue', label='Negative Neurons Count (<-2Z)', fill=True, alpha=0.4)
plt.title("KDE of Synchronized Neurons Count Across All Subjects")
plt.xlabel("Number of Synchronized Neurons")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()