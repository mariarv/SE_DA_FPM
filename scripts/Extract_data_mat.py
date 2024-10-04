import h5py
import pandas as pd
import numpy as np
import pickle

file_path = '/Volumes/T7/optostimulation.mat'

mat_data = h5py.File(file_path, 'r')

# Function to recursively convert HDF5 objects to serializable types
def h5py_to_serializable(obj, file):
    if isinstance(obj, h5py.Dataset):
        data = obj[()]
        if isinstance(data, np.ndarray):
            if data.dtype.kind == 'O':  # Check if it's an object array
                return [h5py_to_serializable(file[obj_ref], file) if isinstance(obj_ref, h5py.Reference) else obj_ref for obj_ref in data]
            elif data.dtype.kind in {'U', 'S'}:  # Handle string arrays
                return data.astype(str).tolist()
            elif data.dtype.kind == 'V':  # Handle structured arrays
                return {name: h5py_to_serializable(data[name], file) for name in data.dtype.names}
            return data.tolist()
        elif isinstance(data, (bytes, np.bytes_)):
            return data.decode('utf-8')  # Convert bytes to string
        else:
            return data
    elif isinstance(obj, h5py.Group):
        return {key: h5py_to_serializable(obj[key], file) for key in obj.keys()}
    elif isinstance(obj, h5py.Reference):
        ref_data = h5py_to_serializable(file[obj], file)
        while isinstance(ref_data, h5py.Reference):
            ref_data = h5py_to_serializable(file[ref_data], file)
        return ref_data
    else:
        return obj

# Function to convert arrays of ASCII values to strings
def ascii_to_string(ascii_values):
    return ''.join(chr(int(value)) for sublist in ascii_values for value in sublist)

# Function to extract string data from references
def extract_string_data(ref_list, file):
    result = []
    for ref in ref_list[0]:
        if isinstance(ref, h5py.Reference):
            ref_data = h5py_to_serializable(file[ref], file)
            while isinstance(ref_data, h5py.Reference):
                ref_data = h5py_to_serializable(file[ref_data], file)
            if isinstance(ref_data, list) and all(isinstance(sublist, list) for sublist in ref_data):
                result.append(ascii_to_string(ref_data))
            elif isinstance(ref_data, (bytes, np.bytes_)):
                result.append(ref_data.decode('utf-8'))
            elif isinstance(ref_data, str):
                result.append(ref_data)
            else:
                result.append(str(ref_data))
        else:
            result.append(str(ref))
    return result

# Function to preprocess the data_dict
def preprocess_data_dict(data_dict, file):
    if 'subject_id' in data_dict:
        data_dict['subject_id'] = extract_string_data(data_dict['subject_id'], file)
    if 'region_id' in data_dict:
        data_dict['region_id'] = extract_string_data(data_dict['region_id'], file)
    if 'region_id' in data_dict:
        data_dict['session'] = extract_string_data(data_dict['session'], file)
    return data_dict

# Function to extract data from the 'opto_combined' structure
def extract_opto_combined(data, file):
    opto_combined = data['opto_combinedVS']
    extracted_data = []

    # Create a dictionary to hold all columns of the dataframe
    data_dict = {key: h5py_to_serializable(opto_combined[key], file) for key in opto_combined.keys()}
    data_dict = preprocess_data_dict(data_dict, file)  # Preprocess the data_dict

    # Convert to a list of dictionaries for each row
    num_rows = len(data_dict['subject_id'])
    for i in range(num_rows):
        row_data = {}
        for key in data_dict.keys():
            if i < len(data_dict[key]):
                if isinstance(data_dict[key], list) or isinstance(data_dict[key], np.ndarray):
                    if isinstance(data_dict[key][i], h5py.Reference):
                        row_data[key] = h5py_to_serializable(file[data_dict[key][i]], file)
                    elif isinstance(data_dict[key][i], (list, np.ndarray)):
                        row_data[key] = [h5py_to_serializable(file[ref], file) if isinstance(ref, h5py.Reference) else ref for ref in data_dict[key][i]]
                        row_data[key] = [item for sublist in row_data[key] for item in (sublist if isinstance(sublist, list) else [sublist])]  # Flatten nested lists
                    else:
                        row_data[key] = data_dict[key][i]
                else:
                    row_data[key] = data_dict[key]
        extracted_data.append(row_data)
    
    return extracted_data

# Function to explode specific columns (e.g., 'data'), while preserving others
def explode_specific_columns(df, columns_to_explode):
    # Explode only the first row for the specified columns
    exploded_dfs = []
    for col in columns_to_explode:
        exploded_col = df[col].apply(pd.Series).stack().reset_index(level=1, drop=True).rename(col)
        exploded_dfs.append(exploded_col.reset_index(drop=True))  # Reset index for exploded_df

    # Create a new DataFrame with the exploded column(s)
    exploded_df = pd.concat(exploded_dfs, axis=1)

    # Extract the other columns that need to be preserved as they are
    other_cols = [col for col in df.columns if col not in columns_to_explode]
    preserved_df = df[other_cols]

    # Repeat the preserved columns' values to match the length of the exploded DataFrame
    repeated_preserved_df = preserved_df.loc[df.index.repeat(df[columns_to_explode[0]].str.len())].reset_index(drop=True)

    # Combine the exploded columns with the preserved columns
    final_df = pd.concat([preserved_df, exploded_df], axis=1)

    return final_df

# Inspect the keys in the .mat file
def inspect_keys(data):
    def print_structure(name, obj):
        print(name)
    data.visititems(print_structure)

print("Inspecting keys in the .mat file...")
inspect_keys(mat_data)

# Extract the "opto_combined" structure
print("Extracting opto_combined data...")
opto_combined_data = extract_opto_combined(mat_data, mat_data)

# Convert the extracted data to a DataFrame
print("Converting to DataFrame...")
df_opto_combined = pd.DataFrame(opto_combined_data)

# Ensure all data in the DataFrame is serializable
def ensure_serializable(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return df

df_opto_combined = ensure_serializable(df_opto_combined)

# Explode only the 'data' column in the DataFrame
print("Exploding specific columns in the DataFrame...")
columns_to_explode = ['data',"duration","session_id","session"]  # Specify which columns to explode
df_opto_combined = explode_specific_columns(df_opto_combined, columns_to_explode)

# Reset the index to ensure no duplicate index values
df_opto_combined.reset_index(drop=True, inplace=True)

# Save the DataFrame to a pickle file
print("Saving DataFrame to a pickle file...")
pickle_filename = 'df_opto_combined_from.pkl'
with open(pickle_filename, 'wb') as f:
    pickle.dump(df_opto_combined, f)
print(f"Saved DataFrame to {pickle_filename}")

# Optionally, display the DataFrame
print("DataFrame for opto_combined:")
print(df_opto_combined.head())
print(df_opto_combined.columns)
