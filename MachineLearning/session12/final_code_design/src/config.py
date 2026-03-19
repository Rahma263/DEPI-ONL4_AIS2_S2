import os

RANDOM_STATE = 45
EXPERIMENT_NAME = "advanced-mlflow-pipeline_George_v3"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
