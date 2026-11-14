"""Model training, evaluation, explainability and the MLflow registry."""

import os

# Silence MLflow's start-up hint for AI coding assistants (log noise only).
# Set here so it applies before any training module imports mlflow.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
