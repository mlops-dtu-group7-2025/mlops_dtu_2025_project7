from omegaconf import OmegaConf

# Define your configuration as a dictionary
config = {
    "paths": {
        "raw_folder": "../../data/raw",
        "train_file": "Train.csv",
        "test_file": "Test.csv",
        "model_output": "lstm_model.pth",
        "predictions_output": "test_predictions.csv",
    },
    "model": {
        "architecture": "LSTM",
        "input_size": 1,
        "hidden_size": 64,
        "num_layers": 2,
    },
    "training": {
        "batch_size": 64,
        "epochs": 20,
        "learning_rate": 0.001,
        "sequence_length": 50,
    },
}

# Convert the dictionary to an OmegaConf object
cfg = OmegaConf.create(config)
