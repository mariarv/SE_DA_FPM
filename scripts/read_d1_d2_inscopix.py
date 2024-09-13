# readin out D1  / D2 miniscope data from mat files

import h5py
import pandas as pd
import numpy as np
import pickle
import os 
import scipy.io 
from scipy.signal import convolve
import metrics_analysis as m_a
file_path = '/Volumes/T7/inscopix data/Saline/d1/'

def h5py_to_serializable(obj):
    if isinstance(obj, h5py.Dataset):
        data = obj[()]
        if isinstance(data, np.ndarray):
            if data.dtype.kind == 'O':  # Object array
                return [h5py_to_serializable(obj_ref) if isinstance(obj_ref, h5py.Reference) else obj_ref for obj_ref in data]
            return data.tolist()
        elif isinstance(data, (bytes, np.bytes_)):
            return data.decode('utf-8')  # Convert bytes to string
        else:
            return data
    elif isinstance(obj, h5py.Group):
        return {key: h5py_to_serializable(obj[key]) for key in obj.keys()}
    else:
        return obj

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    else:
        print(f"File found: {file_path}")
def extract_continuous_data(data):
    dff_data = data['dff_data']
    extracted_data = []

    for i in range(len(dff_data)):
        # Access continuous data
        continuous_data_ref = dff_data['continuous'][i, 0]
        continuous_data = data[continuous_data_ref]['raw']['CNMFE_denoised']
        continuous_data_serialized = h5py_to_serializable(continuous_data)
        
        # Flatten the list of arrays into a single array
        flattened_data = np.concatenate(continuous_data_serialized).ravel()
        extracted_data.append(flattened_data)


    return extracted_data

def convolve_all_traces(traces):
    """Convolve all traces together."""
    if not traces:
        return None

    # Start with the first trace
    convolved_result =   np.sum(traces, axis=0)

    return convolved_result

def process_all_files(directory_path):
    combined_data = []

    for filename in os.listdir(directory_path):
        # Exclude hidden files and non-mat files
        if filename.startswith('.') or not filename.endswith('.mat'):
            continue

        file_path = os.path.join(directory_path, filename)
        check_file_exists(file_path)
        print(f"Processing file: {filename}")
        data = h5py.File(file_path, 'r')
        continuous_data = extract_continuous_data(data)

        # Convolve all traces within the current file
        convolved_data = convolve_all_traces(continuous_data)
        combined_data.append(convolved_data)

    return combined_data

def main(directory_path):
    print("Processing all .mat files in the directory...")
    combined_data = process_all_files(directory_path)
    print("Saving combined data to a pickle file...")
    pickle_filename = 'data/combined_convolved_d1.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump(combined_data, f)
    print(f"Saved combined convolved data to {pickle_filename}")
    print("Sample of combined convolved data:")
    for i, sample in enumerate(combined_data[:3]):
        print(f"Sample {i}: {sample[:10]}... (first 10 elements)")

if __name__ == "__main__":
    main(file_path)