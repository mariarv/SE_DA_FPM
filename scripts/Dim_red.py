import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.signal import resample
from scipy.linalg import hankel, svd
import umap
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA

# Generate chaotic time series using the Lorenz system
def lorenz(t, state, sigma=10.0, beta=8.0/3.0, rho=28.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Function to resample signals
def resample_signal(signal, original_fs, target_fs, target_length):
    num_samples = int(len(signal) * target_fs / original_fs)
    if num_samples != target_length:
        signal = resample(signal, target_length)
    return signal

def create_hankel_matrix(signal, window_size):
    return hankel(signal[:window_size], signal[window_size-200:])

def create_sliding_windows(signal, window_size, step_size):
    num_windows = (len(signal) - window_size) // step_size + 1
    return np.array([signal[i*step_size : i*step_size + window_size] for i in range(num_windows)])


# Sample data (replace with your actual data)
PICKLE_FILE_PATH_DS = 'df_combined_vs_all_drugs.pkl'
ORIGINAL_RATE = 1017.252625
target_fs = 1000  # Target sampling frequency after downsampling
target_segment_length = 200000

df = pd.read_pickle(PICKLE_FILE_PATH_DS)
grouped_signals_before = df
grouped_signals_after = df
# Resample signals and collect in a list
signals_before = []
signals_after = []

window_size = 20000  # Window size for the sliding window
step_size = 200    # Step size for the sliding window

for (signal_idx_before, row_before), (signal_idx_after, row_after) in zip(grouped_signals_before.iterrows(), grouped_signals_after.iterrows()):
    signal_before = resample_signal(row_before['base_before'][2000:144140], ORIGINAL_RATE, target_fs, target_segment_length)
    signal_after = resample_signal(row_after['base_after'][0:142140], ORIGINAL_RATE, target_fs, target_segment_length)
    signal_after=signal_after[0:100000]
    signal_before=signal_before[0:100000]
    n_rows = 2000  # Number of rows in Hankel matrix

    t_span = (0, 50)
    t_eval = np.linspace(0, 50, 500)
    initial_state = [1.0, 1.0, 1.0]
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)

    # Extract the time series from the Lorenz system (using the x-coordinate)
    #chaotic_signal = sol.y[0]

    # Generate noise time series
    np.random.seed(42)  # For reproducibility
    chaotic_signal = np.random.normal(0, 1, len(t_eval))

    H_before = create_sliding_windows(signal_before, window_size, step_size)
    H_after = create_sliding_windows(signal_after, window_size, step_size)

    H_before_flat = H_before.reshape(-1, H_before.shape[1])
    H_after_flat = H_after.reshape(-1, H_after.shape[1])
    time_indices_b = np.arange(len(chaotic_signal)).reshape(-1, 1)
    time_indices_a = np.arange(len(signal_after)).reshape(-1, 1)
    signal_before_reshaped = np.hstack((time_indices_b, chaotic_signal.reshape(-1, 1)))
    signal_after_reshaped = np.hstack((time_indices_a, signal_after.reshape(-1, 1)))
    drug=df["drug"][signal_idx_before]


    # Apply t-SNE on the Hankel matrices with 3 components
    #reducer = umap.UMAP(n_components=3, random_state=42)
   # pca =PCA(n_components=3)
   # umap_before = pca.fit_transform(H_before_flat)
   # umap_after = pca.fit_transform(H_after_flat)
   # umap_before = reducer.fit_transform(H_before_flat)
    #umap_after = reducer.fit_transform(H_after_flat)
    U, S, Vt = np.linalg.svd(H_before_flat, full_matrices=False)

# U contains the spatial modes (eigenvectors)
# S contains the singular values
# Vt contains the temporal coefficients (time evolution of each mode)

# To reconstruct the data using the first few POD modes:
    num_modes = 3  # Number of modes to retain
    umap_before = U
    U_, S_, Vt_ = np.linalg.svd(H_after_flat, full_matrices=False)
    umap_after = U_

    fig = plt.figure(figsize=(14, 12))
    
    # Create 3D subplots for before and after
    ax_before = fig.add_subplot(2, 2, 3, projection='3d')
    ax_after = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Plot the t-SNE results for before
    ax_before.plot(umap_before[:, 0], umap_before[:, 1], umap_before[:, 2],  marker='o', markersize=1, linestyle='-', color='blue')
    ax_before.set_title(f't-SNE Before Signal for {drug}')
    ax_before.set_xlabel('Component 1')
    ax_before.set_ylabel('Component 2')
    ax_before.set_zlabel('Component 3')
    
    # Plot the t-SNE results for after
    ax_after.plot(umap_after[:, 0], umap_after[:, 1], umap_after[:, 2], marker='o', markersize=1, linestyle='-', color='red')
    ax_after.set_title(f't-SNE After Signal {drug}')
    ax_after.set_xlabel('Component 1')
    ax_after.set_ylabel('Component 2')
    ax_after.set_zlabel('Component 3')


    ax_time_before = fig.add_subplot(2, 2, 1)
    ax_time_after = fig.add_subplot(2, 2, 2)
    
    ax_time_before.plot(signal_before, color='blue')
    ax_time_before.set_title(f'Time Series Before Signal for {drug}')
    ax_time_before.set_xlabel('Time')
    ax_time_before.set_ylabel('Amplitude')
    
    ax_time_after.plot(signal_after, color='red')
    ax_time_after.set_title(f'Time Series After Signal for {drug}')
    ax_time_after.set_xlabel('Time')
    ax_time_after.set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()

