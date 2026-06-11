import mlflow
import mlflow.tensorflow


class MLflowTracker:
    def __init__(
        self,
        experiment_name: str,
    ):
        
        mlflow.set_tracking_uri("http://50.19.15.192:5000")
        mlflow.set_experiment(experiment_name)

        

    def start_run(
        self,
        run_name: str,
    ):
        mlflow.start_run(run_name=run_name)

    def end_run(self):
        mlflow.end_run()

    def log_params(
        self,
        params: dict,
    ):
        mlflow.log_params(params)

    def log_metrics(
        self,
        metrics: dict,
    ):
        mlflow.log_metrics(metrics)

    def log_tensorflow_model(
        self,
        model,
        artifact_path: str = "model",
    ):
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path=artifact_path,
        )
