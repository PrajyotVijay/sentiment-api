"""Train a sentiment model on the Amazon reviews dataset and export it to model.pkl."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv("datasets/amazon_dataset.csv").dropna()
X_train, X_test, y_train, y_test = train_test_split(
    df["reviewText"], df["Positive"], test_size=0.2, random_state=42, stratify=df["Positive"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)
print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")

joblib.dump(model, "model.pkl")
print("Saved model.pkl")
