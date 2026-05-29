import mlflow
import mlflow.pytorch

mlflow.set_tracking_uri("file:./mlruns")


def get_experiments():
    return mlflow.search_experiments()


def get_runs(experiment_id):
    return mlflow.search_runs(
        experiment_ids=[experiment_id]
    )


def load_model(run_id):

    model_uri = f"runs:/{run_id}/model"

    return mlflow.pytorch.load_model(
        model_uri
    )