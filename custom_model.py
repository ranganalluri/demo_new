import pandas as pd
import numpy as np
from mlflow.pyfunc import PythonModel
from mlflow.models.signature import infer_signature


class CustomModel(PythonModel):
    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations

    def load_context(self, context):
        print(dir(context.artifacts))
        self.model_path = context.artifacts

    def predict(self, context, model_input):

        results = []
        for _ in range(self.num_simulations):
            simulation_result = self._single_simulation(model_input)
            results.append(simulation_result)

        # Convert results to a DataFrame
        result_df = pd.DataFrame(results)

        # Return the DataFrame directly
        return result_df

    def _single_simulation(self, model_input):
        # Example simulation: sum of random values based on input parameters
        simulation_result = {}
        for param, value in model_input.items():
            simulation_result[param] = np.random.normal(loc=value, scale=1.0)
        return simulation_result

    def get_input_format(self):
        """
        Returns a description of the expected input format.
        """
        return {
            "param1": "Description of param1",
            "param2": "Description of param2",
            # Add descriptions for all required parameters
        }
