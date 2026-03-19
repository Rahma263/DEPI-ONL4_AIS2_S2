import mlflow

if __name__ == "__main__":

    mlflow.set_experiment("testing_mlflow_1")

    with mlflow.start_run(run_name="logging_artifacts") as run:

        mlflow.log_artifacts(local_dir="./run_artifacts", artifact_path="run_artifacts")

        # طباعة معلومات الـ run
        print("run_id:", run.info.run_id)
        print("experiment_id:", run.info.experiment_id)
        print("status:", run.info.status)
        print("start_time:", run.info.start_time)
        print("end_time:", run.info.end_time)
        print("lifecycle_stage:", run.info.lifecycle_stage)