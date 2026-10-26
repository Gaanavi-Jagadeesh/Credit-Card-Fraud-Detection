# Credit Card Fraud Detection — Synthetic Sample Dataset

## Dataset details
- Rows: 5,000 transactions
- Columns: 23
- Target column: `is_fraud` (`0` = non-fraud, `1` = fraud)
- Fraud rows: 88
- Non-fraud rows: 4,912
- Format: CSV and Excel workbook

## Important
This is a **synthetically generated sample dataset** for development, demonstration, and GitHub portfolio use. It is **not the original dataset from Google Colab**, and the records are not real transactions or real cardholder information. Model results trained on this sample should not be presented as real-world performance.

## Files
- `credit_card_fraud_synthetic_5000.csv` — best for Python/VS Code/ML training
- `credit_card_fraud_synthetic_5000.xlsx` — spreadsheet version with a summary sheet

## Loading in Python
```python
import pandas as pd
df = pd.read_csv("credit_card_fraud_synthetic_5000.csv")
print(df.shape)
print(df["is_fraud"].value_counts())
```

## Schema
The columns follow the common transaction dataset layout used in the project notebook. Some fields are identifiers or synthetic demographic/location-like fields and should be assessed carefully before training.
