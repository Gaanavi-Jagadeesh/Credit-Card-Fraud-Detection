"""
Credit Card Fraud Detection — Gradio app
Educational demo for the synthetic 5,000-row dataset.

Keep these files in the same folder:
  app.py
  fraud_model.pkl
  feature_columns.pkl
  credit_card_fraud_synthetic_5000.csv

The saved model should be the fitted preprocessing + Random Forest pipeline
created by fraud.py, so categorical values are passed in as text and numeric
values as numbers.
"""

from pathlib import Path
import warnings

import gradio as gr
import joblib
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_model.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.pkl"
DATASET_PATH = BASE_DIR / "credit_card_fraud_synthetic_5000.csv"


def check_required_files():
    missing = [
        str(path.name)
        for path in (MODEL_PATH, FEATURES_PATH, DATASET_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s): "
            + ", ".join(missing)
            + ". Put these files in the same folder as app.py."
        )


check_required_files()

model = joblib.load(MODEL_PATH)
feature_columns = list(joblib.load(FEATURES_PATH))
sample = pd.read_csv(DATASET_PATH, nrows=5000)

# Treat a column as categorical whenever its actual dataset dtype is not numeric.
# This also handles pandas StringDtype, which may not equal "object".
categorical_columns = []
numeric_columns = []

for col in feature_columns:
    if col not in sample.columns:
        # The training feature list should match the dataset columns.
        raise ValueError(
            f"Feature '{col}' is in feature_columns.pkl but not in "
            f"{DATASET_PATH.name}. Check that the model and dataset are from "
            "the same training run."
        )

    if pd.api.types.is_numeric_dtype(sample[col]):
        numeric_columns.append(col)
    else:
        categorical_columns.append(col)

input_components = []

# Build inputs in the exact feature order used during training.
for col in feature_columns:
    label = col.replace("_", " ").title()

    if col in categorical_columns:
        choices = sorted(sample[col].dropna().astype(str).unique().tolist())
        if not choices:
            choices = ["Unknown"]

        input_components.append(
            gr.Dropdown(
                choices=choices,
                value=choices[0],
                label=label,
                allow_custom_value=False,
            )
        )
    else:
        numeric_values = pd.to_numeric(sample[col], errors="coerce")
        median_value = numeric_values.median()
        default_value = float(median_value) if pd.notna(median_value) else 0.0

        input_components.append(
            gr.Number(
                value=round(default_value, 2),
                label=label,
                precision=4,
            )
        )


def predict_fraud(*values):
    try:
        row = {}

        for col, value in zip(feature_columns, values):
            if col in categorical_columns:
                # Use text for categorical columns; the saved pipeline should
                # apply the same encoder used when training.
                row[col] = str(value) if value is not None else ""
            else:
                if value is None:
                    value = 0
                row[col] = float(value)

        transaction = pd.DataFrame([row], columns=feature_columns)

        prediction = int(model.predict(transaction)[0])

        fraud_probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transaction)[0]
            classes = list(model.classes_)
            if 1 in classes:
                fraud_probability = float(probabilities[classes.index(1)]) * 100.0

        if prediction == 1:
            result = "⚠️ FRAUD PREDICTED"
        else:
            result = "✅ NON-FRAUD PREDICTED"

        probability_text = (
            f"{fraud_probability:.2f}%"
            if fraud_probability is not None
            else "Unavailable for this model"
        )

        return result, probability_text, "Prediction completed successfully."

    except Exception as exc:
        return (
            "Prediction failed",
            "Unavailable",
            f"Error: {type(exc).__name__}: {exc}",
        )


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Credit Card Fraud Detection",
) as demo:
    gr.Markdown(
        """
        # 💳 Credit Card Fraud Detection
        Enter the transaction details and click **Predict Transaction**.

        This is an educational prototype trained on a **synthetic dataset**.
        Its predictions are not suitable for real financial decisions.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Transaction details")
            for component in input_components:
                component.render()
            predict_button = gr.Button("Predict Transaction", variant="primary")

        with gr.Column():
            gr.Markdown("### Result")
            prediction_output = gr.Textbox(label="Prediction", interactive=False)
            probability_output = gr.Textbox(
                label="Model-estimated fraud probability",
                interactive=False,
            )
            status_output = gr.Textbox(label="Status / Error details", interactive=False)

    predict_button.click(
        fn=predict_fraud,
        inputs=input_components,
        outputs=[prediction_output, probability_output, status_output],
    )

    gr.Markdown(
        """
        ---
        **Project note:** The probability is the model's estimate, not a guarantee.
        Synthetic-data performance does not establish real-world fraud detection accuracy.
        """
    )


if __name__ == "__main__":
    demo.launch()
