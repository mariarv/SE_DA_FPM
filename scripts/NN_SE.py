import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from collections import Counter
from imblearn.over_sampling import SMOTE


def generate_sequences(X, y, seq_length, pre_spike_window=10, post_spike_window=990, burst_threshold=20):
    spike_indices = np.where(y == 1)[0]
    sequences = []
    labels = []
    original_start_indices = []

    # Iterate through spike indices
    i = 0
        # Check for burst
    while i < len(spike_indices):
        spike_idx = spike_indices[i]
        start_idx = max(0, spike_idx - pre_spike_window)
        end_idx = min(len(X), spike_idx + post_spike_window)
        label = "tonic"  # Default label

        # Check for burst (considering previous spike)
        if i > 0 and spike_idx - spike_indices[i - 1] <= burst_threshold:
            label = "burst"
            # Extend the sequence to include the entire burst
            while i < len(spike_indices) - 1 and spike_indices[i + 1] - spike_indices[i - 1] <= burst_threshold:
                i += 1
                end_idx = min(len(X), spike_indices[i] + post_spike_window)
        
        seq = X[start_idx:end_idx]

        # Pad sequences if necessary
        if len(seq) < seq_length:
            padding_len = seq_length - len(seq)
            seq = np.pad(seq, (0, padding_len), mode='constant')

        sequences.append(seq)
        labels.append(label)
        original_start_indices.append(start_idx)
        i += 1

    # Generate an equal number of sequences without spikes
    non_spike_indices = np.where(y == 0)[0]
    np.random.shuffle(non_spike_indices)
    non_spike_indices = non_spike_indices[:len(sequences)] 
    #reduced_num_non_spike_sequences = len(sequences) // 2  # Reduce by half
    #non_spike_indices = non_spike_indices[:reduced_num_non_spike_sequences]

    for non_spike_idx in non_spike_indices:
        start_idx = non_spike_idx
        end_idx = min(len(X), non_spike_idx + seq_length)
        seq = X[start_idx:end_idx]

        # Pad if necessary
        if len(seq) < seq_length:
            padding_len = seq_length - len(seq)
            seq = np.pad(seq, (0, padding_len), mode='constant')

        sequences.append(seq)
        labels.append("no spike")
        original_start_indices.append(start_idx)

    # Shuffle all sequences, labels, and original start indices
    max_seq_length = max(len(seq) for seq in sequences)

    # Pad sequences to maximum length
    all_sequences = []
    for seq in sequences:
        all_sequences.append(seq[0:seq_length])

    all_sequences = np.array(all_sequences)
    all_labels = np.array(labels)
    all_original_start_indices = np.array(original_start_indices) 
    
    #shuffle_indices = np.arange(len(all_sequences))
    #all_sequences = all_sequences[shuffle_indices]
    #np.random.shuffle(shuffle_indices)
    #all_labels = all_labels[shuffle_indices]
    #all_original_start_indices = all_original_start_indices[shuffle_indices]

    return all_sequences, all_labels, all_original_start_indices, seq_length

seq_length = 1000  # 250 ms equivalent in points

# Load the ground truth data
ground_truth_df = pd.read_csv('data/ground_truth_data_s.csv')

# Extract features and target
X = ground_truth_df['Smoothed Bulk ΔF/F'].values  
y = ground_truth_df['Bulk Spike Count'].values

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume X and y are already loaded and preprocessed
# Normalize X
#plt.plot(X_sequences)
##plt.plot(y_sequences)
#plt.show()
# Perform the split without shuffling (to preserve time series order)
X = (X - X.mean()) / X.std()
X_sequences, y_sequences, all_original_start_indices, max_seq_length = generate_sequences(X, y, seq_length)

# Reshape for LSTM (assuming 1 feature, which is ΔF/F)
X_sequences = X_sequences.reshape(-1, max_seq_length, 1)  

# Perform the split without shuffling (to preserve time series order)
X_train, X_test, y_train, y_test, original_start_indices_train, original_start_indices_test = train_test_split(
    X_sequences, y_sequences, all_original_start_indices, test_size=0.3, random_state=42, shuffle=False
)
# Convert labels to numerical 
label_mapping = {"no spike": 0, "tonic": 1, "burst": 2}
y_train_sequences = np.array([label_mapping[label] for label in y_train])
y_test_sequences = np.array([label_mapping[label] for label in y_test])
for i in range(3):
    segment = X_train[i]
    label_index = y_train_sequences[i].item()  # Get the numerical label and convert to Python int
    label = [key for key, value in label_mapping.items() if value == label_index][0]  # Map back to string label
    plt.plot(segment)
    plt.show()
    print(f"Segment {i+1} shape: {segment.shape}, Label: {label}")
# Convert to tensors and send to device
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_sequences = torch.tensor(y_train_sequences, dtype=torch.long).to(device)  # LongTensor for CrossEntropyLoss
X_test= torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_sequences = torch.tensor(y_test_sequences, dtype=torch.long).to(device) 


batch_size = 64
train_dataset = TensorDataset(X_train, y_train_sequences)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test_sequences)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 


# Check class distribution in training set
total_label_counts = Counter(y_train)

# Print total percentage of each label
print("Total Label Distribution in Ground Truth Data:")
for label, count in total_label_counts.items():
    percentage = count / len(y_train) * 100
    print(f"  {label}: {count} ({percentage:.2f}%)")
# Define the GRU model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # CNN layers
        x = x.permute(0, 2, 1)  # Permute to (batch_size, channels, seq_length)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, seq_length, channels)

        # LSTM layers
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last output 

        # Fully connected layer
        out = self.fc(out)
        out = self.softmax(out)
        return out

class CNNModel(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, hidden_size, output_size, dropout_prob=0.5):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(num_filters * ((seq_length - 2*(kernel_size-1)) // 2), hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to (batch_size, channels, seq_length)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
# Initialize the model, optimizer, and loss function
input_size = 1  # Assuming one input feature
hidden_size = 128  # Number of units in GRU hidden layers
num_layers = 3  # Number of GRU layers
dropout_prob = 0.3  # Dropout probability
output_size =3
num_filters = 64  # Number of filters in the convolutional layers
kernel_size = 3  # Size of the convolutional kernel

#model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob).to(device)
#model = CNNModel(input_size, num_filters, kernel_size, hidden_size, output_size, dropout_prob).to(device)
# ... (rest of your imports and data preparation)

model = CNNLSTMModel(input_size=input_size, num_filters=num_filters, kernel_size=kernel_size, 
                     hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob).to(device)

# ... (rest of your training and evaluation code)
class_weights = []
count_=[]
for label, count in total_label_counts.items():
    count_.append(count) 

print(count_)
total_count = len(y_train)

class_weights = [total_count / count_[i] if count_[i] > 0 else 1e6 for i in range(len(label_mapping))]


class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Use weighted loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Training loop with overfitting check on a small batch
num_epochs = 30
small_batch_size = 64
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs) 
        # Reshape targets if necessary
        if targets.dim() > 1:
            targets = targets.squeeze()
        # Add an extra dimension to targets

        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}')

# Final model evaluation on test set

torch.save(model, 'results/Spike_infer_model.pth')  # Saves the entire model
y_pred=[]
model.eval()
with torch.no_grad():
    all_outputs = []
    for inputs, _ in test_loader: 
        outputs = model(inputs)

        _, predicted_classes = torch.max(outputs, 1)  # Get the class with highest probability
        y_pred.extend(predicted_classes.cpu().numpy())

        all_outputs.append(outputs)

y_pred_labels = [list(label_mapping.keys())[list(label_mapping.values()).index(pred)] for pred in y_pred]
y_true = y_test_sequences.cpu().numpy().flatten()

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')  # Use 'weighted' for multi-class
recall = recall_score(y_true, y_pred, average='weighted') 

f1 = f1_score(y_true, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}') 