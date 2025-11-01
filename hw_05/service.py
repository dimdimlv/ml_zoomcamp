import pickle
from fastapi import FastAPI
from pydantic import BaseModel


# Load the pipeline at startup
with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

app = FastAPI()


class LeadRecord(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


@app.get("/")
def read_root():
    return {"message": "Lead Scoring API"}


@app.post("/predict")
def predict(record: LeadRecord):
    """
    Predict the probability of a lead converting to a subscription.
    """
    # Convert the Pydantic model to a dictionary
    record_dict = record.model_dump()
    
    # Get the probability (second column is for positive class)
    probability = pipeline.predict_proba([record_dict])[0, 1]
    
    return {
        "probability": float(probability),
        "subscription_likely": bool(probability >= 0.5)
    }
