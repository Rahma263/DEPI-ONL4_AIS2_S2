import mlflow

mlflow.set_experiment("testing_mlflow")  # ينشئ experiment إذا لم يكن موجود

with mlflow.start_run(run_name="logging_artifacts") as run:
    with open("hello_world.txt", "w") as f:
        f.write("Hello World!")
    mlflow.log_artifact("hello_world.txt", "text_files")

    print("run_id:", run.info.run_id)
    print("experiment_id:", run.info.experiment_id)