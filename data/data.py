import os
import requests
import pandas as pd
import typer

import matplotlib.pyplot as plt
import seaborn as sns

from src.mlops_2025_floods_prediction.logging_util import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

app = typer.Typer()

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    try:
        # Load your data here
        logger.debug("Successfully loaded the data.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def download_file(url, folder_path, filename):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses

    # Save the file in the specified folder with the specified filename
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

@app.command()
def download_csv():
    """Download Test.csv and Train.csv files into the 'raw' folder."""
    # Get the absolute path of the folder where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))  

    # Create path to the 'raw' folder in the script's directory
    raw_folder = os.path.join(script_dir, "raw")

    # URLs to download
    test_url = "https://api.zindi.africa/v1/competitions/inundata-mapping-floods-in-south-africa/files/Test.csv?auth_token=zus.v1.mWS8pNR.2d2odFAssmWV1FXtSpgkf9o9E4ZqQh"
    train_url = "https://api.zindi.africa/v1/competitions/inundata-mapping-floods-in-south-africa/files/Train.csv?auth_token=zus.v1.mWS8pNR.2d2odFAssmWV1FXtSpgkf9o9E4ZqQh"
    
    # Download the Test.csv file into the 'raw' folder
    download_file(test_url, raw_folder, "Test.csv")
    
    # Download the Train.csv file into the 'raw' folder
    download_file(train_url, raw_folder, "Train.csv")

@app.command()
def analysis():
    """Analyze the downloaded data and return the number of unique event IDs."""
    # Get the absolute path of the folder where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create path to the 'raw' folder in the script's directory
    raw_folder = os.path.join(script_dir, "raw")

    # File paths
    test_file_path = os.path.join(raw_folder, "Test.csv")
    train_file_path = os.path.join(raw_folder, "Train.csv")

    load_data(train_file_path)

    # Load data
    test_data = pd.read_csv(test_file_path)
    train_data = pd.read_csv(train_file_path)

    # Extract unique event_ids from both datasets
    test_event_ids = test_data['event_id'].apply(lambda x: x.split('_X_')[0]).unique()
    train_event_ids = train_data['event_id'].apply(lambda x: x.split('_X_')[0]).unique()

    # Combine unique event_ids from both datasets
    all_unique_event_ids = set(test_event_ids).union(set(train_event_ids))

    # Count the number of unique event_ids
    test_event_ids_count = len(test_event_ids)
    train_event_ids_count = len(train_event_ids)

    unique_event_count = len(all_unique_event_ids)
    

    # Print the result
    typer.echo(f'Total number of unique event_ids train: {train_event_ids_count}')

    typer.echo(f'Total number of unique event_ids test: {test_event_ids_count}')

    typer.echo(f'Total number of unique event_ids: {unique_event_count}')

@app.command()
def data_inspection():
    # Load the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_folder = os.path.join(script_dir, "raw")
    train_file_path = os.path.join(raw_folder, "Test.csv")

    data = pd.read_csv(train_file_path)

    data['event_id'] = data['event_id'].apply(lambda x: '_'.join(x.split('_')[0:2]))
    data['event_idx'] = data.groupby('event_id', sort=False).ngroup()
    # data_test['event_id'] = data_test['event_id'].apply(lambda x: '_'.join(x.split('_')[0:2]))
    # data_test['event_idx'] = data_test.groupby('event_id', sort=False).ngroup()

    data['event_t'] = data.groupby('event_id').cumcount()
    # data_test['event_t'] = data_test.groupby('event_id').cumcount()

    # Basic exploration
    print(data.head())
    print(data.info())
    print(data.describe())
    print('-------------------------')
    print(data.columns)

if __name__ == "__main__":
    app()
