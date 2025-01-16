import wandb
import os
from datetime import datetime, timedelta

# Initialize W&B API
api = wandb.Api()

# Configuration
entity = os.getenv("WANDB_ENTITY")
project = os.getenv("WANDB_PROJECT")
registry_name = "Floods_prediction"

# Fetch the model registry
try:
    collection = api.artifact_collection(f"{entity}/{project}/{registry_name}")
    latest_artifact = collection.latest()

    # Check if the latest artifact was updated recently (e.g., within the last 30 minutes)
    update_time = latest_artifact.updated_at
    update_time = datetime.strptime(update_time, "%Y-%m-%dT%H:%M:%S.%fZ")

    time_threshold = datetime.utcnow() - timedelta(minutes=30)
    if update_time >= time_threshold:
        print("Model registry has been updated. Triggering further actions...")
        # Add any custom actions here, e.g., notify, retrain, redeploy, etc.
        exit(0)
    else:
        print("No recent updates in the model registry.")
        exit(1)
except Exception as e:
    print(f"Error checking the model registry: {e}")
    exit(1)