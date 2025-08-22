# model_training.py (optimized)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ================================
# 1. Load datasets
# ================================
job_data = pd.read_csv("data/ai_job_market_insights.csv")
salary_data = pd.read_csv("data/salaries.csv")

# Rename to align column names
job_data.rename(columns={"Job_Title": "job_title"}, inplace=True)

# Merge datasets
df = pd.merge(job_data, salary_data, on="job_title", how="inner")

# Drop missing values
df = df.dropna()

# ================================
# 2. Reduce dataset size (to save memory)
# ================================
df = df.sample(n=20000, random_state=42)  # adjust if memory allows

# Target column
target_column = "Industry"

# Separate features (X) and labels (y)
X = df.drop(columns=[target_column])
y = df[target_column]

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X, drop_first=True)

# ================================
# 3. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 4. Model Training (Random Forest)
# ================================
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# ================================
# 5. Evaluation
# ================================
y_pred = rf.predict(X_test)

print("📌 Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nRandom Forest Report:\n", classification_report(y_test, y_pred))

# ================================
# 6. Save Model
# ================================
os.makedirs("models", exist_ok=True)

# Save trained Random Forest model
joblib.dump(rf, "models/ai_career_model.pkl")

# Save feature names used during training
joblib.dump(X_encoded.columns.tolist(), "models/feature_columns.pkl")

print("✅ Random Forest model and feature columns saved!")
