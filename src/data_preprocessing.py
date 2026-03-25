import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import DATA_PATH, TEST_SIZE, RANDOM_STATE

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess_data(df):
    # Separate features & target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    print("Before SMOTE:")
    print(y.value_counts())

    # Handle imbalance using SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)

    print("\nAfter SMOTE:")
    print(pd.Series(y_res).value_counts())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test