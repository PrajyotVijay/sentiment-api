# Review Sentiment API

A FastAPI service that predicts whether a product review is **positive or negative**.
Model: TF-IDF + Logistic Regression (scikit-learn) trained on 20,000 Amazon reviews.
Built for the *Getting Started with ML in Production* workshop (IIT Madras).

**Live API:** <PASTE YOUR RENDER URL HERE>  (docs at `/docs`)

## What the model predicts
Given review text, it returns `positive` or `negative` with a confidence score.

## Example request for `/predict`
```json
{
  "text": "Great game, my kids love it and the levels are really fun"
}
```
Example response:
```json
{
  "prediction": 1,
  "sentiment": "positive",
  "confidence": 0.992
}
```

## Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Status and whether the model is loaded |
| POST | `/predict` | Takes review text, returns sentiment |

## Run it locally
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py                 # creates model.pkl
uvicorn main:app --reload
```
Then open http://127.0.0.1:8000/docs or test with curl:
```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Great game, my kids love it and the levels are really fun"}'
```

## Deploy on Render
Build command: `pip install -r requirements.txt` · Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
