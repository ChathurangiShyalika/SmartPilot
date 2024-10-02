from sktime.utils import mlflow_sktime
import os

os.environ["MLFLOW_SKTIME_USE_GPU"] = "true"

def predictAnomaly(TSCInput):
    loaded_model = mlflow_sktime.load_model("model")

    pred = loaded_model.predict(TSCInput)
    manual_mapping = {'Two Body Parts and Nose Cone Missing': 3, 'Top Body and Nose Cone Missing': 2, 'Nose Cone Missing': 1, 'Normal': 0}
    reverse_mapping = {v: k for k, v in manual_mapping.items()}
    categorical_predictions = [reverse_mapping[label] for label in pred]
    print(categorical_predictions)
    return categorical_predictions

