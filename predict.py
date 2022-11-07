import bentoml
from bentoml.io import JSON
import numpy as np

model_ref = bentoml.xgboost.get("price_prediction_model:latest")

dv = model_ref.custom_objects["preprocessor"]

model_runner = model_ref.to_runner()

svc = bentoml.Service("airbnb_price_predictor", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def predict_price(application_data):
    vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)[0]
    predict_price =   np.exp(prediction) - 1       
    return {"predicted_price": predict_price}