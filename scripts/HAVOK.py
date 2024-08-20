import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd
from sklearn.linear_model import LinearRegression
from scipy.signal import resample
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt
from scipy.stats import boxcox

def resample_signal(signal, original_fs, target_fs, target_length):
    num_samples = int(len(signal) * target_fs / original_fs)
    if num_samples != target_length:
        signal = resample(signal, target_length)
    return signal

def create_hankel_matrix(data, n_rows, n_cols):
    return hankel(data[:n_rows], data[n_rows-1:n_rows+n_cols-1])

def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def havok_analysis(signal, title, r):
    n_rows = 60  # Number of rows in Hankel matrix
    n_cols = len(signal) # Number of columns in Hankel matrix
    H = create_hankel_matrix(signal, n_rows, n_cols)
    U, s, Vt = svd(H, full_matrices=False)
    
    # Truncate the SVD Matrices
    Ur = U[:, :r]
    Sr = np.diag(s[:r])
    Vr = Vt[:r, :]
    
    # Compute the Derivative of Vr
    dt =1.5
    dVr = np.gradient(Vr, dt, axis=1)
    
    # Linear Regression to Discover the Dynamics
    Vr_combined =Vr[:, :-1]
    dVr_combined = dVr[:, :-1]
    
    model = LinearRegression(fit_intercept=False)
    model.fit(Vr_combined.T, dVr_combined.T)
    A = model.coef_.T
    
    # Reconstruct Dynamics and Validate
    def koopman_dynamics(t, y, A):
        return A @ y
    
    y0 = Vr[:, 0]
    t_span = (0, Vr.shape[1] - 1)
    t_eval = np.arange(Vr.shape[1])
    
    sol = solve_ivp(koopman_dynamics, t_span, y0, t_eval=t_eval, args=(A,))
    
    reconstructed_signal_normalized = Ur @ Sr @ sol.y

    # Plot the original signal and reconstructed signal
    plt.figure(figsize=(10, 5))
    plt.plot(signal, label='Original Signal')
    plt.plot(np.arange(len(reconstructed_signal_normalized[0, :])), reconstructed_signal_normalized[0, :], label='Reconstructed Signal')
    plt.legend()
    plt.title(f"Original and Reconstructed Signals using HAVOK - {title}")
    plt.show()

    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    for i in range(4):
        axs[i].plot(Vr[i, :])
        axs[i].set_title(f"Component V{i + 1}")
    plt.tight_layout()
    plt.show()

    return A, Vr, dVr, sol


def plot_attractor(signal_before, signal_after, title_before, title_after, delay=50, dimension=10):
    def time_delay_embedding(data, delay, dimension):
        N = len(data)
        M = N - (dimension - 1) * delay
        if M <= 0:
            raise ValueError("The length of the data is too short for the given delay and embedding dimension.")
        embedded_data = np.zeros((M, dimension))
        for i in range(dimension):
            embedded_data[:, i] = data[i * delay:M + i * delay]
        return embedded_data

    embedded_before = time_delay_embedding(signal_before, delay, dimension)
    embedded_after = time_delay_embedding(signal_after, delay, dimension)
    
    fig = plt.figure(figsize=(14, 7))
    
    ax_before = fig.add_subplot(121, projection='3d')
    ax_before.plot(embedded_before[:, 0], embedded_before[:, 1], embedded_before[:, 2])
    ax_before.set_title(title_before)
    ax_before.set_xlabel('X(t)')
    ax_before.set_ylabel('X(t + τ)')
    ax_before.set_zlabel('X(t + 2τ)')
    
    ax_after = fig.add_subplot(122, projection='3d')
    ax_after.plot(embedded_after[:, 0], embedded_after[:, 1], embedded_after[:, 2])
    ax_after.set_title(title_after)
    ax_after.set_xlabel('X(t)')
    ax_after.set_ylabel('X(t + τ)')
    ax_after.set_zlabel('X(t + 2τ)')
    
    plt.show()

# Load the dataframe
PICKLE_FILE_PATH_DS = 'df_combined_ds_all_drugs.pkl'
ORIGINAL_RATE = 1017.252625
target_fs = 1000  # Target sampling frequency after downsampling
target_segment_length = 10000  # Target length after resampling
# Extract the "after_coc" column
df = pd.read_pickle(PICKLE_FILE_PATH_DS)
df['file_prefix'] = df['file'].apply(lambda x: x.split('_')[0])
grouped_signals_before = df.groupby('file_prefix')['base_before'].apply(lambda x: np.concatenate(x.values)).reset_index()
grouped_signals_after = df.groupby('file_prefix')['base_after'].apply(lambda x: np.concatenate(x.values)).reset_index()

for (signal_idx_before, row_before), (signal_idx_after, row_after) in zip(grouped_signals_before.iterrows(), grouped_signals_after.iterrows()):
    # Extract the signal segments
    signal_before = row_before['base_before']
    signal_after = row_after['base_after']
    signal_after = low_pass_filter(signal_after, 5, ORIGINAL_RATE)
    signal_before = low_pass_filter(signal_before, 5, ORIGINAL_RATE)
    signal_before = resample_signal(row_before['base_before'][2000:144140], ORIGINAL_RATE, target_fs, target_segment_length)
    signal_after = resample_signal(row_after['base_after'][0:142140], ORIGINAL_RATE, target_fs, target_segment_length)
    # Perform EMD on the "before" signal
    # Perform HAVOK analysis on the "before" signal
    signal_before=signal_before[0:1000]
    signal_after=signal_after[0:1000]
    shifted_signal = signal_after - np.min(signal_after) + 1
    signal_after, _ = boxcox(shifted_signal)
    
    shifted_signal_ = signal_before - np.min(signal_before) + 1
    signal_before, _ = boxcox(shifted_signal_)
    A_before, Vr_before, dVr_before, sol_before = havok_analysis(signal_before, f"Before Signal {signal_idx_before}",r=20)

    # Perform HAVOK analysis on the "after" signal
    A_after, Vr_after, dVr_after, sol_after = havok_analysis(signal_after, f"After Signal {signal_idx_after}",r=30)

    # Compare Koopman matrices
    print(f"Koopman matrix for 'Before' Signal {signal_idx_before}:\n{A_before}")
    print(f"Koopman matrix for 'After' Signal {signal_idx_after}:\n{A_after}")

    # Visualize attractors
    plot_attractor(signal_before, signal_after, f"Attractor - Before Signal {signal_idx_before}", f"Attractor - After Signal {signal_idx_after}")