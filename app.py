from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model (Pipeline: StandardScaler + SVM)
model = joblib.load("model.pkl")

# Mapping from model class (bulk) to human-readable sea condition
SEA_CONDITION_LABELS = {
    0: "Small / almost flat",
    1: "Medium, surfable waves",
    2: "Larger / more powerful waves",
}


def recommend_activity(condition_id: int, windspeed_ms: float) -> dict:
    """
    Translate wave-size class (bulk) + windspeed (in m/s)
    into a human-readable activity recommendation for WavePredictor.
    """

    # Define simple wind thresholds (approx):
    # low   <= 5 m/s  (~10 kts)
    # medium 5–9 m/s (~10–18 kts)
    # high  >= 9 m/s (~18+ kts)
    low_wind = 5.0
    high_wind = 9.0

    # Class 0: small / almost flat
    if condition_id == 0:
        if windspeed_ms >= high_wind:
            # Flat sea + strong wind = ideal for wind sports
            return {
                "activity": "Kitesurf / Windsurf 🪁",
                "message": (
                    "Waves are very small, but the wind is strong. "
                    "Great for kitesurfing or windsurfing, not really a surfing day."
                ),
            }
        else:
            # Calm sea + low/medium wind = swim & SUP
            return {
                "activity": "Swimming & SUP 🏊‍♂️",
                "message": (
                    "Small, calm sea. Perfect for swimming, relaxing or stand-up paddling, "
                    "but not really for classic surfing."
                ),
            }

    # Class 1: medium, typically surfable
    if condition_id == 1:
        if windspeed_ms <= low_wind:
            return {
                "activity": "Surfing 🏄‍♀️",
                "message": (
                    "Medium-sized waves and low wind – nice, regular surf conditions "
                    "for most skill levels."
                ),
            }
        elif windspeed_ms < high_wind:
            return {
                "activity": "Surfing (wind-affected) 🌊",
                "message": (
                    "Medium-sized waves with some wind. Still surfable, but conditions "
                    "might be choppy or less clean."
                ),
            }
        else:
            return {
                "activity": "Choppy surf / wind sports",
                "message": (
                    "Medium waves but strong wind. Good for experienced surfers "
                    "or for mixing in some wind sports."
                ),
            }

    # Class 2: larger / powerful waves
    if condition_id == 2:
        if windspeed_ms <= low_wind:
            return {
                "activity": "Advanced surfing only ⚠️",
                "message": (
                    "Larger, more powerful waves with low wind. "
                    "Good for advanced surfers, beginners should be very careful."
                ),
            }
        elif windspeed_ms < high_wind:
            return {
                "activity": "Advanced surfing / challenging conditions ⚠️",
                "message": (
                    "Bigger waves and noticeable wind. Fun for advanced surfers, "
                    "but demanding and not beginner-friendly."
                ),
            }
        else:
            return {
                "activity": "Mostly wind sports / very demanding surf 🪁",
                "message": (
                    "Strong wind and larger waves. This is more a day for kitesurfing "
                    "or windsurfing; surfing will be very heavy and messy."
                ),
            }

    return {
        "activity": "Unknown",
        "message": "The sea condition is outside the expected range.",
    }


@app.route("/")
def index():
    """Render main HTML page for WavePredictor."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict wave-size class (bulk) and return activity recommendation."""
    try:
        data = request.get_json()

        # Read values from request (UI sends windspeed in knots)
        sigheight = float(data["sigheight"])
        swellheight = float(data["swellheight"])
        period = float(data["period"])
        windspeed_knots = float(data["windspeed"])
        winddirdegree = float(data["winddirdegree"])

        # Convert knots -> m/s because the model was trained on m/s
        # 1 knot ≈ 0.514444 m/s
        windspeed_ms = windspeed_knots * 0.514444

        # Features must follow the same order as during training
        features = np.array(
            [[sigheight, swellheight, period, windspeed_ms, winddirdegree]]
        )

        # Model prediction
        bulk_class = int(model.predict(features)[0])
        sea_condition = SEA_CONDITION_LABELS.get(bulk_class, "Unknown")

        # Business logic: translate into human activity
        rec = recommend_activity(bulk_class, windspeed_ms)

        return jsonify(
            {
                "bulk_class": bulk_class,
                "sea_condition": sea_condition,
                "activity": rec["activity"],
                "message": rec["message"],
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5500)