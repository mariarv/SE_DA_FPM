import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from scipy.signal import resample
import logging
import os
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Paths to the pickle files
PICKLE_FILE_PATH_VS = 'data/df_combined_vs.pkl'
PICKLE_FILE_PATH_DS = 'data/df_combined_ds.pkl'
TEMPLATE_FILE_PATH = 'data/Templates/After_Cocaine_DS_mean_traces_dff.csv'
ORIGINAL_RATE = 1017.252625
TARGET_RATE = 100  # Example target rate
BATCH_SIZE = 16  # Number of traces to process at a time
NUM_THREADS = 4  # Number of threads for parallel processing
SHUFFLE_BUFFER_SIZE = 100  # Shuffle buffer size to reduce memory usage
EPOCHS = 3  # Number of epochs for training

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataframe(pickle_file_path):
    if not os.path.exists(pickle_file_path):
        logging.error(f"File not found: {pickle_file_path}")
        raise FileNotFoundError(f"File not found: {pickle_file_path}")
    with open(pickle_file_path, 'rb') as f:
        df = pickle.load(f)
    logging.info(f"Loaded data from {pickle_file_path}")
    return df

def load_template(template_file_path):
    if not os.path.exists(template_file_path):
        logging.error(f"File not found: {template_file_path}")
        raise FileNotFoundError(f"File not found: {template_file_path}")
    template_ = pd.read_csv(template_file_path)
    template=template_["25"].values
    logging.info(f"Loaded template from {template_file_path}")
    return np.array(template)

def resample_data(data, original_rate, target_rate):
    num_samples = int(len(data) * target_rate / original_rate)
    resampled_data = resample(data, num_samples)
    return resampled_data

def preprocess_data(data, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    else:
        data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()
    return data_scaled, scaler

def create_training_data(traces, template, window_size):
    X = []
    y = []
    template_length = len(template)
    
    for trace in traces:
        for i in range(len(trace) - window_size + 1):
            window = trace[i:i + window_size]
            X.append(window)
            
            if i <= len(trace) - template_length:
                match = trace[i:i + template_length]
                label = 1 if np.allclose(match, template, atol=0.1) else 0
            else:
                label = 0
            y.append(label)
    
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    return X, y

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))  # Adding dropout for regularization
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))  # Adding dropout for regularization
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def trace_generator(df, batch_size, original_rate, target_rate, template, scaler=None):
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        traces = [np.array(row['base_after_coc']) for _, row in batch_df.iterrows() if len(row['base_after_coc']) > 0]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(resample_data, trace, original_rate, target_rate) for trace in traces]
            resampled_traces = [future.result() for future in futures]
        
        scaled_traces = []
        for trace in resampled_traces:
            scaled_trace, scaler = preprocess_data(trace, scaler)
            scaled_traces.append(scaled_trace)
        
        X, y = create_training_data(scaled_traces, template, len(template))
        yield X, y

def main(pickle_file_path_vs, pickle_file_path_ds, template_file_path, original_rate, target_rate, batch_size, epochs):
    try:
        
        #df_vs = load_dataframe(pickle_file_path_vs)
        df_ds = load_dataframe(pickle_file_path_ds)[["base_before_coc"]]
        template = load_template(template_file_path)

        #traces_vs = [np.array(row['base_after_coc']) for _, row in df_vs.iterrows() if len(row['base_after_coc']) > 0]
        traces_ds = [np.array(row['base_after_coc']) for _, row in df_ds.iterrows() if len(row['base_after_coc']) > 0]
        #all_traces = traces_vs + traces_ds

        resampled_template = resample_data(template, original_rate, target_rate)
        scaled_template, scaler = preprocess_data(resampled_template)

        model = build_model((len(scaled_template), 1))
        # Split data into training and validation sets
        train_df, val_df = train_test_split(df_ds, test_size=0.2, random_state=42)

        # Initialize lists to store training and validation metrics
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        # Use a data generator to load data in batches from the DataFrame
        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch + 1}/{epochs}")
            train_loss, train_accuracy = [], []
            for traces_batch in trace_generator(train_df, batch_size, original_rate, target_rate, scaled_template, scaler):
                train_dataset = tf.data.Dataset.from_tensor_slices((traces_batch[0], traces_batch[1])).batch(batch_size)
                train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE)
                
                metrics = model.fit(train_dataset, epochs=1, steps_per_epoch=len(train_dataset))
                train_loss.append(metrics.history['loss'][0])
                train_accuracy.append(metrics.history['accuracy'][0])
            
            history['train_loss'].append(np.mean(train_loss))
            history['train_accuracy'].append(np.mean(train_accuracy))

            # Validate the model
            val_loss, val_accuracy = [], []
            for traces_batch in trace_generator(val_df, batch_size, original_rate, target_rate, scaled_template, scaler):
                val_dataset = tf.data.Dataset.from_tensor_slices((traces_batch[0], traces_batch[1])).batch(batch_size)
                loss, accuracy = model.evaluate(val_dataset)
                val_loss.append(loss)
                val_accuracy.append(accuracy)
            
            history['val_loss'].append(np.mean(val_loss))
            history['val_accuracy'].append(np.mean(val_accuracy))
        
        # Plot training and validation accuracy and loss
        epochs_range = range(1, epochs + 1)
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_loss'], label='Training Loss')
        plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['train_accuracy'], label='Training Accuracy')
        plt.plot(epochs_range, history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        
        plt.show()

        logging.info("Training completed")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main(PICKLE_FILE_PATH_VS, PICKLE_FILE_PATH_DS, TEMPLATE_FILE_PATH, ORIGINAL_RATE, TARGET_RATE, BATCH_SIZE, EPOCHS)
