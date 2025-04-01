from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import uvicorn

# Import the helper function to load the model
from src.utils.helpers import load_model
from src.data_processing import prepare_data_for_production
from src.models.xgboost import predict_xgboost

app = FastAPI(title="Fraud Detection API", description="API for predicting fraudulent transactions.")

# Define a Pydantic model for incoming prediction requests.
class PredictionRequest(BaseModel):
    transaction_table: Dict[str, Any]
    identity_table: Dict[str, Any]

# Load the model at startup.
model = load_model("xgboost_model")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fraud Detection API. Use POST /predict to get a prediction."}

@app.get("/health")
def health_check():
    """
    Simple health-check endpoint.
    """
    return {"status": "ok", "message": "API is healthy!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert the incoming JSON features to a DataFrame.
        # The input should be a dictionary mapping feature names to values.
        df_transaction = pd.DataFrame([request.transaction_table])
        df_identity = pd.DataFrame([request.identity_table])
        input_df = prepare_data_for_production(df_transaction, df_identity, categorical_handling='object_to_category')

        prediction = predict_xgboost(model, input_df)
        
        # Return the predicted probability (or label, if thresholding is applied).
        return {"prediction": float(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
