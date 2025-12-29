import pandas as pd
import joblib
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("data/songs.csv")


median_streams = df["streams"].median()
df["hit"] = (df["streams"] >= median_streams).astype(int)


FEATURES = ["danceability", "energy", "loudness", "tempo", "valence"]
X = df[FEATURES]
y = df["hit"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=38
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=100, random_state=38)

lr.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)


lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled))

best_model = rf if rf_acc > lr_acc else lr
best_model_name = "Random Forest" if rf_acc > lr_acc else "Logistic Regression"


os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/best_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

metrics = {
    "total_samples": len(df),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "best_model": best_model_name,
    "comparison": {
        "Logistic Regression": {
            "accuracy": lr_acc,
            "type": "Linear Classification",
            "strengths": "Fast, simple, interpretable",
            "weaknesses": "Cannot model complex patterns"
        },
        "Random Forest": {
            "accuracy": rf_acc,
            "type": "Ensemble Tree-Based",
            "strengths": "Handles non-linearity, high accuracy",
            "weaknesses": "Less interpretable, slower"
        }
    }
}

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Training complete.")

print("Logistic Regression Accuracy:", lr_acc)
print("Random Forest Accuracy:", rf_acc)
print("Best Model:", best_model_name)
