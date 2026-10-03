"""FastAPI app serving a sentiment model (TF-IDF + Logistic Regression).

Run locally:  uvicorn main:app --reload
"""
import logging
import os
import time

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sentiment-api")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(title="Review Sentiment API", version="1.0.0")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except FileNotFoundError:
    model = None
    logger.error("model.pkl not found - run train.py first")


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="A product review")


class PredictionResponse(BaseModel):
    prediction: int
    sentiment: str
    confidence: float


@app.get("/")
def root():
    return {"message": "Sentiment API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.perf_counter()
    proba = model.predict_proba([request.text])[0]
    pred = int(proba.argmax())
    result = PredictionResponse(
        prediction=pred,
        sentiment="positive" if pred == 1 else "negative",
        confidence=round(float(proba[pred]), 4),
    )
    logger.info("predict | sentiment=%s confidence=%.3f latency_ms=%.1f",
                result.sentiment, result.confidence, (time.perf_counter() - start) * 1000)
    return result
