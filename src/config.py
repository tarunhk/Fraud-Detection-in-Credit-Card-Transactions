DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/xgb_model.pkl"

TEST_SIZE = 0.2
RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1
}