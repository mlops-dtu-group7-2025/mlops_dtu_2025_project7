import wandb

# Authenticate with W&B
wandb.login()

# Replace "your-project-name" and "model-registry-name" with actual names
project_name = "mlops_dtu_2025_project7"  # Update this with your W&B project name
model_registry_name = "model-registry"  # Replace with the actual model registry artifact name

# Log the model registry changes
try:
    run = wandb.init(project=project_name, job_type="model-update")
    
    # Use the model registry artifact
    artifact = run.use_artifact(f"{model_registry_name}:latest")
    
    # Add metadata or additional context
    artifact.metadata["updated_by"] = "GitHub Actions"
    artifact.metadata["update_reason"] = "Triggered by GitHub push to model registry"

    # Save the changes
    artifact.save()
    print("Triggered W&B workflow for model registry update.")

except Exception as e:
    print(f"Error triggering W&B workflow: {e}")
finally:
    run.finish()