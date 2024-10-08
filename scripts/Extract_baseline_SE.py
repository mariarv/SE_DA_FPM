import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Path to the pickle file created in the previous version
pickle_file_path = 'data/df_combined_SE_row.pkl'

# Function to calculate time vector
def calculate_time_vector(start, end, rate, length):
    return np.linspace(start, end, length)

# Function to extract base traces from the DataFrame
def extract_base_traces(df):
    extracted_data = []
    for index, row in df.iterrows():
        ep2_timestamps = np.array(row['Ep2_timestamps'])
        epopto_timestamps = np.array(row['Epopto_timestamps'])
        
        bl_rec_data = np.array(row['bl_rec_data'])
        bl_rec_samplerate = row['bl_rec_samplerate']
        bl_rec_tstart = row['bl_rec_tstart']
        bl_rec_tend = row['bl_rec_tend']
        
        post_cocaine_bl_data = np.array(row['post_cocaine_bl_data'])

        file_value = row['file']

        # Calculate time vectors
        bl_rec_time_vector = calculate_time_vector(bl_rec_tstart, bl_rec_tend, bl_rec_samplerate, len(bl_rec_data))
        
        # Extract base_before_coc
        ep2_start = ep2_timestamps[0]
        epopto_end_candidates = epopto_timestamps[epopto_timestamps < ep2_start]
        if epopto_end_candidates.size > 0:
            epopto_end = epopto_end_candidates[-1]
            base_before_coc_indices = (bl_rec_time_vector <= ep2_start) & (bl_rec_time_vector >= epopto_end)
            base_before_coc = bl_rec_data[base_before_coc_indices]
        else:
            base_before_coc = np.array([])

        # Extract base_after_coc
        ep2_end = ep2_timestamps[-1]
        ep2_end_plus_5min = ep2_end + 5 * 60  # Adding 5 minutes
        epopto_start_candidates = epopto_timestamps[epopto_timestamps > ep2_end]
        if epopto_start_candidates.size > 0:
            epopto_start = epopto_start_candidates[0]
            base_after_coc_indices = (bl_rec_time_vector >= ep2_end_plus_5min) & (bl_rec_time_vector <= epopto_start)
            base_after_coc = post_cocaine_bl_data[base_after_coc_indices]
        else:
            base_after_coc = np.array([])

        row_data = {
            'file': file_value,
            'base_before_coc': base_before_coc,
            'base_after_coc': base_after_coc
        }
        extracted_data.append(row_data)
    
    return pd.DataFrame(extracted_data)

# Function to ensure all data in the DataFrame is serializable
def ensure_serializable(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return df

# Function to split the DataFrame based on the presence of 'DS' or 'VS' in the 'file' column
def split_dataframe(df):
    df_ds = df[df['file'].str.contains('DS')]
    df_vs = df[df['file'].str.contains('VS')]
    return df_ds, df_vs

# Function to plot data
def plot_data(df, condition, trace_type):
    plt.figure(figsize=(12, 6))
    for index, row in df.head(5).iterrows():
        if trace_type == 'base_before_coc':
            plt.plot(row[trace_type], label=f"{row['file']} - {condition}")
        elif trace_type == 'base_after_coc':
            plt.plot(row[trace_type], label=f"{row['file']} - {condition}")
    
    plt.title(f"All {trace_type.replace('_', ' ')} traces for {condition}")
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.show()


# Main function to load the pickle file and process data
def main(pickle_file_path):
    # Load the DataFrame from the pickle file
    print("Loading DataFrame from pickle file...")
    with open(pickle_file_path, 'rb') as f:
        df_combined = pickle.load(f)
    
    # Extract the base traces
    print("Extracting base traces...")
    df_combined = extract_base_traces(df_combined)
    
    # Split the DataFrame based on 'DS' or 'VS' in the 'file' column
    print("Splitting the DataFrame...")
    df_ds, df_vs = split_dataframe(df_combined)
    
    # Plot data
    print("Plotting data...")
    plot_data(df_vs, 'VS', 'base_before_coc')
    plot_data(df_vs, 'VS', 'base_after_coc')
    plot_data(df_ds, 'DS', 'base_before_coc')
    plot_data(df_ds, 'DS', 'base_after_coc')
    
    # Save the DataFrames to new pickle files
    print("Saving DataFrames to pickle files...")
    pickle_filename_ds = 'df_combined_ds.pkl'
    with open(pickle_filename_ds, 'wb') as f:
        pickle.dump(df_ds, f)
    print(f"Saved DS DataFrame to {pickle_filename_ds}")
    
    pickle_filename_vs = 'df_combined_vs.pkl'
    with open(pickle_filename_vs, 'wb') as f:
        pickle.dump(df_vs, f)
    print(f"Saved VS DataFrame to {pickle_filename_vs}")

    # Optionally, display the DataFrames
    print("DS DataFrame:")
    print(df_ds.head())
    print("VS DataFrame:")
    print(df_vs.head())

# Run the main function
if __name__ == "__main__":
    main(pickle_file_path)
