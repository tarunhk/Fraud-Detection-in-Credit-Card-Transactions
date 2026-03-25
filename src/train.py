from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import joblib

from data_preprocessing import load_data, preprocess_data
from config import MODEL_PATH, XGB_PARAMS

# Load & preprocess data
df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train XGBoost model
print("\nTraining XGBoost...")
xgb_model = XGBClassifier(**XGB_PARAMS)
xgb_model.fit(X_train, y_train)

# Train Isolation Forest (unsupervised)
print("Training Isolation Forest...")
iso_model = IsolationForest(contamination=0.01)
iso_model.fit(X_train)

# Save models
joblib.dump(xgb_model, MODEL_PATH)
joblib.dump(iso_model, "models/isolation_forest.pkl")

print("Models saved successfully!")