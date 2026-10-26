[README.md](https://github.com/user-attachments/files/33980873/README.md)
# 💳 Credit Card Fraud Detection

A machine learning project that predicts whether a credit card transaction may be fraudulent. It combines a trained Random Forest classifier with a simple Gradio interface, allowing users to enter transaction details and receive a prediction.

> **Educational project:** This version uses a synthetic dataset. It is a demonstration of a machine-learning workflow, not a financial fraud-prevention system.

## ✨ Features

- Data loading and basic inspection with **Pandas**
- Fraud classification using **Random Forest**
- Model evaluation with accuracy, precision, recall, confusion matrix, and classification report
- Saved model and feature-column files for reuse
- Interactive prediction interface built with **Gradio**
- Reproducible local setup using a Python virtual environment

## 🧰 Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Gradio

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
├── fraud.py
├── app.py
├── credit_card_fraud_synthetic_5000.csv
├── credit_card_fraud_synthetic_5000.xlsx
├── fraud_model.pkl
├── feature_columns.pkl
├── README.md
└── .gitignore
```

The exact files in your repository may differ. The `.venv/` folder should stay local and must not be committed to GitHub.

## ⚙️ How It Works

1. Load the transaction dataset.
2. Separate the target label (`is_fraud`) from the input features.
3. Split the data into training and test sets.
4. Train a Random Forest classifier.
5. Evaluate its predictions on the held-out test set.
6. Save the trained model and feature names.
7. Run the Gradio app to submit transaction details and view the prediction.

## 📊 Example Model Results

The current local run reported the following results on a test set of 1,000 transactions:

| Metric | Result |
|---|---:|
| Accuracy | 84.30% |
| Precision for fraud class | 5.16% |
| Recall for fraud class | 44.44% |
| Fraud cases detected | 8 of 18 |
| False positives | 147 |

**How to interpret this:** Accuracy alone is not enough for fraud detection, especially when fraudulent transactions are uncommon. The reported precision is low, so many transactions flagged as fraud were actually legitimate. These are results from one run and should be re-evaluated after any changes to the data or model.

## 🚀 Getting Started

### 1. Clone the repository

Replace the URL below with your repository URL if you fork or rename the project.

```bash
git clone https://github.com/Gaanavi-Jagadeesh/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn joblib gradio openpyxl
```

### 4. Train the model

```bash
python fraud.py
```

This should create `fraud_model.pkl` and `feature_columns.pkl` if the training script completes successfully.

### 5. Launch the app

```bash
python app.py
```

Open the local URL printed in the terminal (usually a Gradio address such as `http://127.0.0.1:7860`).

> **Important:** The app requires the saved model, feature-column file, and dataset at the paths expected by `app.py`. The feature columns and preprocessing used by the app must match those used to train the model.

## 🗂️ Dataset

The project uses a synthetic dataset containing 5,000 transactions, with `is_fraud` as the target label in the training script. Synthetic data is useful for demonstrating a workflow, but its patterns may not represent real-world fraud.

## 🔐 Data and Model Safety

- Do not upload real cardholder data, personal information, credentials, or API keys.
- Only load `.pkl` files from sources you trust.
- Keep your local `.venv/` directory out of version control.
- The saved model file is a generated artifact; if it is missing, run the training script before launching the app.

## 🔭 Future Improvements

- Compare Random Forest with Logistic Regression and XGBoost.
- Improve fraud-class precision and recall.
- Evaluate class-imbalance techniques and decision thresholds.
- Add plots for the confusion matrix and precision-recall curve.
- Add automated tests and clearer input validation.
- Validate on a suitable, documented dataset before considering real-world use.

## 👩‍💻 Author

**Gaanavi Jagadeesh**

A machine-learning project exploring fraud classification, model evaluation, and an interactive prediction application.

---

If you find this project useful, feel free to star the repository ⭐
