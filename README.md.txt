# Billboard Hit Predictor

This project predicts whether a song will become a Billboard hit using machine learning.

## Models Used
- Logistic Regression
- Random Forest Classifier

## Best Model
Random Forest achieved the highest accuracy.

## How to Run

1. Install dependencies
```bash
pip install -r requirements.txt
2.Train the model
python train_model.py
3.Run the app
python main.py
What the Project Does
Loads song data from data/songs.csv containing Spotify audio features and streaming data

Creates hit labels based on median streams:

Hit = streams ≥ median streams

Non-hit = streams < median streams

Uses Spotify audio features for prediction:

Danceability

Energy

Loudness

Tempo

Valence

Trains two models:

Logistic Regression

Random Forest

Evaluates model performance using accuracy scores and compares both models

Saves the best model and scaler for future predictions

Machine Learning Models Used
Logistic Regression Classifier

Random Forest Classifier

Output
The project predicts:

Hit (1): Song is likely to become a Billboard hit

Non-hit (0): Song is unlikely to become a Billboard hit

Saved Files
After training, the following files are saved:

model/best_model.pkl (Best performing model)

model/scaler.pkl (Feature scaler)

model/metrics.json (Training metrics and model comparison)

These files can be reused to make predictions without retraining.

Goal
To build a reliable song hit prediction system using Spotify audio features and streaming data.