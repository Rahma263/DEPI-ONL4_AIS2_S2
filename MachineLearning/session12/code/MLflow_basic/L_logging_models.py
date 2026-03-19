import mlflow
import mlflow.sklearn
from mlflow_utils import get_mlflow_experiment

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import matplotlib
matplotlib.use("Agg")  # backend بدون واجهة رسومية

if __name__ == "__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow_1")
    
    # التحقق من وجود experiment
    if experiment is None:
        raise Exception("Experiment not found. Please create it first.")

    print("Experiment Name: {}".format(experiment.name))

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="logging_models", experiment_id=experiment.experiment_id) as run:

        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=5,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=43
        )

        rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        rfc.fit(X_train, y_train)

        mlflow.sklearn.log_model(rfc, artifact_path="random_forest_classifier")

        print("run_id:", run.info.run_id)
        print("experiment_id:", run.info.experiment_id)
        print("status:", run.info.status)
        print("start_time:", run.info.start_time)
        print("end_time:", run.info.end_time)
        print("lifecycle_stage:", run.info.lifecycle_stage)