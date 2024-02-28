import uvicorn  # ASGI

from inputFeatureExtract import CustomerSegmentationDataLoader
import pandas as pd
from fastapi import FastAPI, Body, HTTPException
from preprocessor import Preprocessor
import pickle
import logging
import json

# Create FastAPI app instance
app = FastAPI()

# logger_app for logging
logger_app = logging.getLogger("app")

# Load purchase predictor model and dependencies
with open("purchase_predictor_model.pkl", "rb") as pickled_model:
    purchase_predictor = pickle.load(pickled_model)

with open("trained_imputer_values.json", "r") as imputer_values:
    mean_values = json.load(imputer_values)

with open("scaler_state.pkl", "rb") as f:
    scaler_state = pickle.load(f)


@app.get("/purchase")
def index():
    return {"message": "Hello"}


# Expose the prediction functionality, make a prediction from the passed json data,
# and return the predicted probability of the customer to purchase an item.
@app.post("/predict")
def predict(raw_data: CustomerSegmentationDataLoader = Body(...)):

    final_features = list(purchase_predictor.params.index)
    preprocessor = Preprocessor()
    data_df = preprocessor.json_to_dataframe(raw_data.features)
    print(data_df)
    logger_app.info("Processing data...")

    try:
        test_imputed_std = preprocessor.preprocess_data(
            data_df, final_features, mean_values, scaler_state
        )
        outcomes_df = pd.DataFrame(purchase_predictor.predict(test_imputed_std[final_features])) \
            .rename(columns={0: "probs"})
        print(outcomes_df.to_json(orient="records"))
        logger_app.info("Prediction successful!")
        return outcomes_df.to_json(orient="records")
    except Exception as e:
        logger_app.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")


# Run the API with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # uvicorn app:app --reload --port=1313 --host=0.0.0.0

