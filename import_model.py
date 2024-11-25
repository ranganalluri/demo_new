import os
from custom_model import CustomModel
import mlflow
import pandas as pd
import numpy as np
from mlflow.models.signature import infer_signature

from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv("dev.env")

mlflow.set_registry_uri("databricks-uc")
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")

os.environ["MLFLOW_TRACKING_URI"] = "databricks"
os.environ["MLFLOW_TRACKING_TOKEN"] = os.getenv("DATABRICKS_TOKEN")


client = MlflowClient()

# Specify the experiment name and (optional) artifact location
experiment_name = "/Users/ranga.nalluri@hotmail.com/custommodel-new"
artifact_location = "dbfs:/Users/ranga.nalluri@hotmail.com/custommodel-new"

exp = client.get_experiment_by_name(experiment_name)
if exp is None:
    client.create_experiment(experiment_name, artifact_location)


mlflow.set_experiment(experiment_name)


with mlflow.start_run() as run:
    # mlflow.log_artifact("C:/mlruns/920401203747447845/91b1d0429c48491f80cea37a0812c041")
    input_format = CustomModel().get_input_format()
    model = CustomModel(num_simulations=1000)
    example_input = {"param1": 0.5, "param2": 3.0}
    example_output = model.predict(None, example_input)
    signature = infer_signature(pd.DataFrame([example_input]), example_output)

    mlflow.pyfunc.log_model(
        artifact_path="custom_model",
        python_model=model,
        registered_model_name="mlflow_ops.default.custommodel",
        signature=signature,
        input_example=example_input,
    )

    mlflow.log_param("num_simulations", 1000)
    mlflow.log_param("input_format", model.get_input_format())

    mlflow.log_metric("metric_test", 0.85)
