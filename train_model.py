import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def main():
    # 1. Load prepared dataset
    df = pd.read_csv("data/sea_activity_dataset.csv")

    # 2. Define features and target
    feature_cols = ["sigheight", "swellheight", "period", "windspeed", "winddirdegree"]
    target_col = "bulk"

    X = df[feature_cols]
    y = df[target_col].astype(int)

    # 3. Train/test split (stratified because classes are imbalanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. Build pipeline: scaling + SVM with RBF kernel
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )

    # 5. Train the model
    print("Training SVM model on real sea forecast data...")
    model.fit(X_train, y_train)

    # 6. Evaluate on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on test set: {acc * 100:.2f}%")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 7. Save model for later use in Flask app
    joblib.dump(model, "model.pkl")
    print("\nSaved trained model to model.pkl")


if __name__ == "__main__":
    main()