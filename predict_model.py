import os
import mlflow
import pandas as pd
import numpy as np
from dotenv import load_dotenv


load_dotenv("dev.env")

mlflow.set_registry_uri("databricks-uc")
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")


model_uri = "models:/mlflow_ops.default.custommodel/4"

model = mlflow.pyfunc.load_model(model_uri)

for i in model.metadata.signature.inputs:
    print(i.name, i.type, i.required)

output = model.predict(pd.DataFrame([{"param1": 100.0, "param2": 3.0}]))
print(output)