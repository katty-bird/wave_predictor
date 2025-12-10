import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib

# Load data and model
df = pd.read_csv("data/sea_activity_dataset.csv")
feature_cols = ["sigheight", "swellheight", "period", "windspeed", "winddirdegree"]
X = df[feature_cols]
y = df["bulk"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load("model.pkl")
y_pred = model.predict(X_test)

cmat = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

n_classes = cmat.shape[0]

diag_vals = cmat[np.eye(n_classes, dtype=bool)]
off_vals = cmat[~np.eye(n_classes, dtype=bool)]

diag_max = diag_vals.max() if diag_vals.size > 0 else 1
off_max = off_vals.max() if off_vals.size > 0 else 1

green_cmap = cm.get_cmap("Greens")
orange_cmap = cm.get_cmap("YlOrBr")

color_matrix = np.zeros((n_classes, n_classes, 4))

for i in range(n_classes):
    for j in range(n_classes):
        value = cmat[i, j]
        if i == j:
            # correct predictions → always clearly green
            norm_v = value / diag_max if diag_max > 0 else 0.0
            # clamp into [0.5, 0.9] so that:
            # - the smallest diagonal cell is still visibly green
            # - the largest one does not become too dark
            norm_v = 0.5 + 0.4 * norm_v
            color_matrix[i, j] = green_cmap(norm_v)
        else:
            # misclassifications → yellow/orange, avoid dark brown tones
            norm_v = value / off_max if off_max > 0 else 0.0
            # clamp into [0.3, 0.8] → from light yellow to soft orange
            norm_v = 0.3 + 0.5 * norm_v
            color_matrix[i, j] = orange_cmap(norm_v)

plt.figure(figsize=(4.5, 4.5))
plt.imshow(color_matrix, interpolation="nearest")
plt.title("Confusion matrix (SVM, bulk prediction)")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.xticks([0, 1, 2])
plt.yticks([0, 1, 2])

# add raw counts into each cell
for i in range(n_classes):
    for j in range(n_classes):
        value = cmat[i, j]
        plt.text(j, i, str(value), ha="center", va="center",
                 color="black", fontsize=10)

plt.text(
    -0.5,
    n_classes + 0.2,
    "Green diagonal = correct predictions\nYellow cells = misclassifications",
    fontsize=8,
)

plt.tight_layout()
plt.show()