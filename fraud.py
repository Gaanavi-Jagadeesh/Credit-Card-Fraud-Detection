import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# 1. Load the dataset from the same folder as this Python file.
DATASET_PATH = "credit_card_fraud_synthetic_5000.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded:", df.shape)

# 2. Remove identifier and high-cardinality/location columns.
drop_cols = [
    "Unnamed: 0", "trans_date_trans_time", "unix_time",
    "first", "last", "street", "city", "state", "zip",
    "lat", "long", "dob", "merch_lat", "merch_long",
    "city_pop", "trans_num", "cc_num",
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 3. Separate features and target.
target_col = "is_fraud"
if target_col not in df.columns:
    raise ValueError("Dataset must contain an 'is_fraud' target column.")

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

print("Target counts:")
print(y.value_counts())

# 4. Preprocess categorical and numerical columns.
categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("numeric", "passthrough", numeric_cols),
    ]
)

# 5. Build and train a Random Forest pipeline.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# 6. Evaluate the model.
y_pred = model.predict(X_test)
print("\n========== MODEL PERFORMANCE ==========")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# 7. Save the trained pipeline and feature names for the app.
joblib.dump(model, "fraud_model.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\nModel training complete!")
print("Saved: fraud_model.pkl")
print("Saved: feature_columns.pkl")