#tkae only "after" and find fearuere that unite them in one class of a drug 

import pandas as pd
import numpy as np
from scipy.signal import resample
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pywt
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw

# Function to resample signals
def resample_signal(signal, original_fs, target_fs, target_length):
    num_samples = int(len(signal) * target_fs / original_fs)
    if num_samples != target_length:
        signal = resample(signal, target_length)
    return signal

def extract_fourier_features(signal):
    fft_values = np.fft.fft(signal)
    magnitude = np.abs(fft_values)
    return magnitude[:len(magnitude) // 2]

# Load data
PICKLE_FILE_PATH_DS = 'df_combined_vs_all_drugs.pkl'
ORIGINAL_RATE = 1017.252625
target_fs = 500  # Target sampling frequency after downsampling
target_segment_length = 300000

# Initialize lists to store signals and labels
signals = []
labels = []
conditions = []

df = pd.read_pickle(PICKLE_FILE_PATH_DS)
grouped_signals_before = df
grouped_signals_after = df
signals_before = []
signals_after = []
ground_truth = []

# Iterate over each pair of before and after signals
for idx, ((signal_idx_before, row_before), (signal_idx_after, row_after)) in enumerate(zip(grouped_signals_before.iterrows(), grouped_signals_after.iterrows())):
    # Extract the signals
    drug=df["drug"][signal_idx_before]
    signal_before = row_before['base_before']
    signal_after = row_after['base_after']
    signal_before = resample_signal(row_before['base_before'][1000:344140], ORIGINAL_RATE, target_fs, target_segment_length)
    signal_after = resample_signal(row_after['base_after'][0:342140], ORIGINAL_RATE, target_fs, target_segment_length)
    signal_after=signal_after[5000:50000]
    signal_before=signal_before[5000:50000]
   


    signals.append(signal_after)
    labels.append(f'{drug} - After')
    conditions.append('After')
    ground_truth.append(drug)  # Assuming there is a column 'class_label' for ground truth


# Convert to numpy array and scale
signals = np.array(signals)
signals = np.array(signals)
signals=signals.T
qt = QuantileTransformer()
signals_transformed = qt.fit_transform(signals)

#signals_scaled = np.array([standardize_trace(trace) for trace in signals_transformed])

# Apply StandardScaler
scaler = StandardScaler()
signals_scaled = scaler.fit_transform(signals)
signals_scaled=signals_scaled.T
ground_truth = np.array(ground_truth)

# Feature extraction


# Perform DTW-based K-Means clustering
model = TimeSeriesKMeans(n_clusters=9, metric="dtw", max_iter=10, random_state=42)
clusters = model.fit_predict(signals_scaled)
label_encoder = LabelEncoder()
ground_truth_encoded = label_encoder.fit_transform(ground_truth)

# Evaluation Metrics
ari = adjusted_rand_score(ground_truth_encoded, clusters)
nmi = normalized_mutual_info_score(ground_truth_encoded, clusters)

# Confusion Matrix
conf_matrix = confusion_matrix(ground_truth_encoded, clusters)

print(f'Adjusted Rand Index (ARI): {ari}')
print(f'Normalized Mutual Information (NMI): {nmi}')
print('Confusion Matrix:')
print(conf_matrix)
