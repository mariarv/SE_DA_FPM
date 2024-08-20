import h5py
import pandas as pd
import numpy as np
import pickle
import os 
import scipy.io 

file_path = 'data/Matlab_files/Se_drugs.mat'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    try:
        # Try to open the file with h5py (HDF5 format)
        mat_data = h5py.File(file_path, 'r')
        is_hdf5_format = True
        print("File opened with h5py.")
    except OSError:
        # If it fails, try to open the file with scipy.io (MATLAB v7.2 or older format)
        mat_data = scipy.io.loadmat(file_path)
        is_hdf5_format = False
        print("File opened with scipy.io.")

# Function to check if the file exists
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    else:
        print(f"File found: {file_path}")

# Function to convert MATLAB cell arrays to serializable lists
def convert_matlab_cell_to_list(cell_array):
    return [cell_array[i][0] for i in range(cell_array.size)]

# Function to extract combined data from the .mat file
def extract_combined_data(data):
    num_rows = data['combined'].size  # Number of rows in Ep2.timestamps

    extracted_data = []
    for i in range(num_rows):
        combined = data['combined'][i]
        row_data = {
            'Ep2_timestamps': combined['Ep2'][0][0][0][5].flatten(),
            'file': combined['file'][0][0],
            'bl_rec_data': combined['bl_rec'][0][0][0][1].flatten(),
            'bl_rec_samplerate': combined['bl_rec'][0][0][0][4][0][0],
            'bl_rec_tstart':combined['bl_rec'][0][0][0][5][0][0],
            'bl_rec_tend': combined['bl_rec'][0][0][0][6][0][0],
            'Epopto_timestamps': combined['Epopto'][0][0][0][5].flatten(),
            'post_cocaine_bl_data': combined['post_cocaine_bl'][0][0][0][1].flatten()
        }
        extracted_data.append(row_data)
    
    return extracted_data

# Function to ensure all data in the DataFrame is serializable
def ensure_serializable(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return df

# Main function to load the .mat file and extract data
def main(file_path):
    check_file_exists(file_path)
    
    # Load the .mat file using scipy.io
    data = scipy.io.loadmat(file_path)
    
    # Extract the combined data
    print("Extracting combined data...")
    combined_data = extract_combined_data(data)
    
    # Convert the extracted data to a DataFrame
    print("Converting to DataFrame...")
    df_combined = pd.DataFrame(combined_data)
    
    # Ensure all data in the DataFrame is serializable
    df_combined = ensure_serializable(df_combined)
    df_combined.reset_index(drop=True, inplace=True)
    
    # Save the DataFrame to a pickle file
    print("Saving DataFrame to a pickle file...")
    pickle_filename = 'data/df_combined_SE.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump(df_combined, f)
    print(f"Saved DataFrame to {pickle_filename}")

    # Optionally, display the DataFrame
    print("DataFrame for combined:")
    print(df_combined.head())

# Run the main function
if __name__ == "__main__":
    main(file_path)