import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score

# Get the absolute path of the folder where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create path to the 'raw' folder in the script's directory
raw_folder = os.path.abspath(os.path.join(script_dir, "../../data/raw"))

# File paths
test_file_path = os.path.join(raw_folder, "Test.csv")
train_file_path = os.path.join(raw_folder, "Train.csv")

# Load data
test_data = pd.read_csv(test_file_path)
train_data = pd.read_csv(train_file_path)

train_data['event_id'] = train_data['event_id'].apply(lambda x: '_'.join(x.split('_')[0:2]))
train_data['event_idx'] = train_data.groupby('event_id', sort=False).ngroup()
test_data['event_id'] = test_data['event_id'].apply(lambda x: '_'.join(x.split('_')[0:2]))
test_data['event_idx'] = test_data.groupby('event_id', sort=False).ngroup()

train_data['event_t'] = train_data.groupby('event_id').cumcount()
test_data['event_t'] = test_data.groupby('event_id').cumcount()

# Group by event_id and sort by time step
train_df = train_data.sort_values(by=["event_id", "event_t"])
test_df = test_data.sort_values(by=["event_id", "event_t"])

# Create sequences for training data
sequence_length = 50  # Choose sequence length based on task requirements
X_train, y_train = [], []

for event_id, group in train_df.groupby("event_id"):
    precip_values = group["precipitation"].values
    labels = group["label"].values
    
    # Generate sliding windows for sequences
    for i in range(len(precip_values) - sequence_length):
        X_train.append(precip_values[i:i + sequence_length])
        y_train.append(labels[i + sequence_length - 1])  # Predict flood label for last step

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Normalize precipitation values using Min-Max Scaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)

print(f"Training sequences shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")


# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Create DataLoader
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take output of last timestep
        x = self.fc(x)
        return self.sigmoid(x)

model = LSTMModel()
criterion = nn.BCELoss()  # Binary classification loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training Loop
epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch.unsqueeze(-1))  # Add feature dimension
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")


# Predictions on training data (for demonstration)
# model.eval()
# with torch.no_grad():
#     y_pred = model(X_train_tensor.unsqueeze(-1)).squeeze().numpy()
#     y_pred_class = (y_pred > 0.5).astype(int)

# print(f"Training Accuracy: {accuracy_score(y_train, y_pred_class):.4f}")
# print(f"ROC-AUC Score: {roc_auc_score(y_train, y_pred):.4f}")


# Prepare test sequences similarly to train sequences
X_test = []

for event_id, group in test_df.groupby("event_id"):
    precip_values = group["precipitation"].values
    
    # Generate sliding windows for sequences
    for i in range(len(precip_values) - sequence_length):
        X_test.append(precip_values[i:i + sequence_length])

# Normalize test data
X_test = np.array(X_test)
X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

# Convert to tensor and predict
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor.unsqueeze(-1)).squeeze().numpy()

# Save predictions
test_df = test_df.iloc[:len(test_predictions)]
test_df['flood_probability'] = test_predictions
test_df.to_csv("test_predictions.csv", index=False)
print("Predictions saved to test_predictions.csv")

