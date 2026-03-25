import shap
import joblib
import matplotlib.pyplot as plt

from data_preprocessing import load_data, preprocess_data
from config import MODEL_PATH

# Load model
model = joblib.load(MODEL_PATH)

# Load data
df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)

# Use small sample for speed
X_sample = X_test[:200]

# SHAP Explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_sample)

# Summary plot
shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig("outputs/shap_summary.png")
plt.close()

print("SHAP summary plot saved!")

# Feature Importance Plot
import matplotlib.pyplot as plt

importances = model.feature_importances_

plt.figure()
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.savefig("outputs/feature_importance.png")
plt.close()

print("Feature importance plot saved!")