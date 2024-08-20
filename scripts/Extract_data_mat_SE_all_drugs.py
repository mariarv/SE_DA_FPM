import h5py
import pandas as pd
import numpy as np
import pickle
import os 
import scipy.io 

file_path = '/Users/reva/Documents/MATLAB/for maria/drugeffects_.mat'

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

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    else:
        print(f"File found: {file_path}")

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

def ascii_to_string(ascii_values):
    return ''.join(chr(int(value)) for sublist in ascii_values for value in sublist)

def preprocess_data_dict(data_dict, file):
    if 'file' in data_dict:
        data_dict['file'] = extract_string_data(data_dict['file'], file)
    if 'session' in data_dict:
        data_dict['session'] = extract_string_data(data_dict['session'], file)
    return data_dict

def extract_combined_data(data, file):
    opto_combined = data['combined']
    extracted_data = []

    # Create a dictionary to hold all columns of the dataframe
    data_dict = {key: h5py_to_serializable(opto_combined[key], file) for key in opto_combined.keys()}
    data_dict = preprocess_data_dict(data_dict, file)  # Preprocess the data_dict

    # Convert to a list of dictionaries for each row
    num_rows = len(data_dict['session'])
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

def filter_extracted_data(data_):
    filtered_data = []
    data=data_[0]
    for ind,entry in enumerate(data_):
        filtered_entry = {
            'file': entry['file'],
            'session': entry['session']
        }

        # Process Ep2 timestamps
        filtered_entry['Ep2_timestamps'] = data['Ep2'][ind]['timestamps'][0]

        filtered_entry['bl_rec_data'] = data['bl_rec'][ind]['data'][0]
        filtered_entry['bl_rec_samplerate'] = data['bl_rec'][ind]['samplerate'][0]
        filtered_entry['bl_rec_tstart'] =data['bl_rec'][ind]['tstart'][0]
        filtered_entry['bl_rec_tend'] = data['bl_rec'][ind]['tend'][0]

        # Consolidate Epopto timestamps
        filtered_entry['Epopto_timestamps'] = data['Epopto'][ind]['timestamps'][0]


        filtered_data.append(filtered_entry)
    
    return filtered_data


def ensure_serializable(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return df

def main(file_path):
    check_file_exists(file_path)
    data = h5py.File(file_path, 'r')
    print("Extracting combined data...")
    combined_data = extract_combined_data(data, data)
    print("Filtering extracted data...")
    filtered_data = filter_extracted_data(combined_data)
    print("Converting to DataFrame...")
    df_combined = pd.DataFrame(filtered_data)
    df_combined = ensure_serializable(df_combined)
    df_combined.reset_index(drop=True, inplace=True)
    print("Saving DataFrame to a pickle file...")
    pickle_filename = 'data/df_combined_SE_all_drugs.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump(df_combined, f)
    print(f"Saved DataFrame to {pickle_filename}")
    print("DataFrame for combined:")
    print(df_combined.head())

if __name__ == "__main__":
    main(file_path)
