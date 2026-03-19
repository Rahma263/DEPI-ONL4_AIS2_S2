import mlflow

if __name__ == "__main__":

    mlflow.set_experiment("testing_mlflow_1")

    parameters = {
        "learning_rate": 0.01,
        "epochs": 10,
        "batch_size": 100,
        "loss_function": "mse",
        "optimizer": "adam"
    }

    with mlflow.start_run(run_name="logging_params") as run:

        mlflow.log_params(parameters)

        print("run_id:", run.info.run_id)
        print("experiment_id:", run.info.experiment_id)