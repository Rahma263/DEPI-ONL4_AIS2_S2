import os
import csv
import json
from datetime import datetime
import mlflow

# --------------------------------------------------------
# Universal Tracker: can log to both MLflow and local file
# --------------------------------------------------------

def log_experiment(run_name, params, metrics, artifacts=None, use_mlflow=True, output_dir="outputs"):
    """

    """
    os.makedirs(output_dir, exist_ok=True)
    logs_path = os.path.join(output_dir, "logs.csv")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- MLflow logging ---
    if use_mlflow:
        mlflow.set_experiment("advanced-mlflow-pipeline")

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Log params
            if params:
                mlflow.log_params(params)
            # Log metrics
            if metrics:
                mlflow.log_metrics(metrics)
            # Log artifacts
            if artifacts:
                for key, path in artifacts.items():
                    if path and os.path.exists(path):
                        mlflow.log_artifact(path, artifact_path="plots")

            # Also store run_id in local CSV
            _append_to_csv(logs_path, run_name, timestamp, run_id, params, metrics)

            print(f"✅ Experiment logged successfully in MLflow (run_id={run_id})")

    else:
        # --- Local CSV logging ---
        _append_to_csv(logs_path, run_name, timestamp, "LOCAL", params, metrics)
        print(f"📁 Experiment logged locally to {logs_path}")


def _append_to_csv(logs_path, run_name, timestamp, run_id, params, metrics):
    """
    helper function to append to local CSV log
    """
    header = ["timestamp", "run_name", "run_id", "params", "metrics"]

    if not os.path.exists(logs_path):
        with open(logs_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(logs_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            run_name,
            run_id,
            json.dumps(params),
            json.dumps(metrics)
        ])
