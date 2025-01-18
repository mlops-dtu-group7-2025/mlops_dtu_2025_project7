import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import wandb
import hydra
from omegaconf import DictConfig

from logging_util import setup_logging
import logging

# Setup logging
setup_logging()

# Create a logger for this module
logger = logging.getLogger(__name__)

logger.info("Starting the training script.")
logger.debug("Debug information about the dataset.")
logger.warning("Potential issue detected.")
logger.error("An error occurred.")
logger.critical("Critical issue! Immediate attention required.")

# Hydra config decorator
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize W&B using Hydra config
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config={
            "architecture": cfg.model.architecture,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
            "learning_rate": cfg.training.learning_rate,
            "sequence_length": cfg.training.sequence_length,
            "hidden_size": cfg.model.hidden_size,
        }
    )

    # Get the absolute path of the folder where the script is located
    script_dir = os.getcwd()
    raw_folder = os.path.join(script_dir, cfg.paths.raw_folder)

    # File paths
    train_file_path = os.path.join(raw_folder, cfg.paths.train_file)
    test_file_path = os.path.join(raw_folder, cfg.paths.test_file)

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
    sequence_length = cfg.training.sequence_length
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
    batch_size = cfg.training.batch_size
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

    # Define LSTM Model
    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=cfg.model.hidden_size, num_layers=cfg.model.num_layers, batch_first=True)
            self.fc = nn.Linear(cfg.model.hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x, _ = self.lstm(x)
            x = x[:, -1, :]  # Output of the last timestep
            x = self.fc(x)
            return self.sigmoid(x)

    model = LSTMModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Training Loop
    epochs = cfg.training.epochs

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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

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
    test_df.to_csv(cfg.paths.predictions_output, index=False)
    print(f"Predictions saved to {cfg.paths.predictions_output}")

    # Log predictions to W&B
    wandb.save(cfg.paths.predictions_output)

    # Save the model
    torch.save(model.state_dict(), cfg.paths.model_output)
    wandb.save(cfg.paths.model_output)

    # Finish W&B logging
    wandb.finish()
if __name__ == "__main__":
    main()