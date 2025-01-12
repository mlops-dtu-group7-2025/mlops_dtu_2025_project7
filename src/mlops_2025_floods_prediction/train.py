import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import wandb

# Initialize a new W&B run
wandb.init(
    project="floods-prediction",  # Set the project name
    config={
        "architecture": "LSTM",
        "dataset": "Flood Prediction",
        "batch_size": 64,
        "epochs": 20,
        "learning_rate": 0.001,
    }
)

# Get the absolute path of the folder where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_folder = os.path.abspath(os.path.join(script_dir, "../../data/raw"))

# File paths
test_file_path = os.path.join(raw_folder, "Test.csv")
train_file_path = os.path.join(raw_folder, "Train.csv")

# Load data
test_data = pd.read_csv(test_file_path)
train_data = pd.read_csv(train_file_path)

# Process event IDs and timestamps
train_data['event_id'] = train_data['event_id'].apply(lambda x: '_'.join(x.split('_')[:2]))
train_data['event_idx'] = train_data.groupby('event_id', sort=False).ngroup()
test_data['event_id'] = test_data['event_id'].apply(lambda x: '_'.join(x.split('_')[:2]))
test_data['event_idx'] = test_data.groupby('event_id', sort=False).ngroup()

train_data['event_t'] = train_data.groupby('event_id').cumcount()
test_data['event_t'] = test_data.groupby('event_id').cumcount()

# Sort data by event and time
train_df = train_data.sort_values(by=["event_id", "event_t"])
test_df = test_data.sort_values(by=["event_id", "event_t"])

# Create sequences for training
sequence_length = 50
X_train, y_train = [], []

for event_id, group in train_df.groupby("event_id"):
    precip_values = group["precipitation"].values
    labels = group["label"].values
    
    for i in range(len(precip_values) - sequence_length):
        X_train.append(precip_values[i:i + sequence_length])
        y_train.append(labels[i + sequence_length - 1])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Normalize precipitation values
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)

print(f"Training sequences shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Create DataLoader
batch_size = wandb.config["batch_size"]
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
        x = x[:, -1, :]  # Output of the last timestep
        x = self.fc(x)
        return self.sigmoid(x)

model = LSTMModel()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])

# Training Loop
epochs = wandb.config["epochs"]

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch.unsqueeze(-1))
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Prepare test sequences
X_test = []

for event_id, group in test_df.groupby("event_id"):
    precip_values = group["precipitation"].values
    
    for i in range(len(precip_values) - sequence_length):
        X_test.append(precip_values[i:i + sequence_length])

X_test = np.array(X_test)
X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

# Predict on test data
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
model.eval()

with torch.no_grad():
    test_predictions = model(X_test_tensor.unsqueeze(-1)).squeeze().numpy()

# Save predictions
test_df = test_df.iloc[:len(test_predictions)]
test_df['flood_probability'] = test_predictions
test_df.to_csv("test_predictions.csv", index=False)
print("Predictions saved to test_predictions.csv")

# Log predictions to WandB
wandb.save("test_predictions.csv")

# Save the model
torch.save(model.state_dict(), "lstm_model.pth")
wandb.save("lstm_model.pth")

# Finish W&B logging
wandb.finish()