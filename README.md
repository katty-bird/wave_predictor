# WavePredictor 🌊

WavePredictor is a small end-to-end ML project that predicts a **wave-size class** from sea forecast data and turns it into a short activity suggestion (surfing, swimming/SUP, kitesurfing, …).

The project is purely educational and focuses on showing how to connect a trained model with a small web interface.

---

## What it does

- Takes typical forecast inputs (significant wave height, swell height, period, wind speed, wind direction).
- Uses a trained SVM model to predict a **wave-size class** (0 = small/flat, 1 = medium/surfable, 2 = larger/powerful).
- Adds a small rule-based layer that combines the predicted class with wind speed to suggest an activity for the spot (e.g. “Swimming & SUP”, “Surfing”, “Kitesurf/Windsurf”, “Advanced surfing only”).

The web UI is intentionally simple and is meant for educational purposes, not as a replacement for real surf forecast services.

---

## Data

WavePredictor is based on the public Kaggle dataset  
[**Sea Forecast and Waves Classification**](https://www.kaggle.com/datasets/saurabhshahane/sea-forecast-and-waves-classification/data).

---

## Deployment

The app is deployed on Render as a simple Flask App
[**WavePredictor**](https://wave-predictor.onrender.com/).