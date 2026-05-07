
import json
import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)

# ── 1. Load & Clean ─────────────────────────────────────────────────────────
print("Loading dataset...")
try:
    df = pd.read_csv("dataset.csv")
except FileNotFoundError:
    # Generate synthetic data matching the notebook schema if CSV not present
    print("  dataset.csv not found — generating synthetic data for demo...")
    np.random.seed(42)
    n = 5000
    genders = ["Male", "Female"]
    occupations = ["Engineer", "Doctor", "Teacher", "Nurse", "Manager",
                   "Accountant", "Lawyer", "Sales Representative", "Scientist", "Software Engineer"]
    bmi_cats = ["Normal", "Overweight", "Obese"]
    exercise_types = ["Cardio", "Strength", "Yoga", "None"]
    disorders = ["None", "Insomnia", "Sleep Apnea"]

    df = pd.DataFrame({
        "Age": np.random.randint(20, 65, n),
        "Gender": np.random.choice(genders, n),
        "Occupation": np.random.choice(occupations, n),
        "BMI_Category": np.random.choice(bmi_cats, n, p=[0.5, 0.35, 0.15]),
        "Smoking_Status": np.random.choice(["Yes", "No"], n, p=[0.2, 0.8]),
        "Sleep_Duration_Hours": np.round(np.random.normal(7, 1.2, n).clip(3, 10), 1),
        "Physical_Activity_Mins": np.random.randint(0, 120, n),
        "Stress_Level": np.random.randint(1, 11, n),
        "Heart_Rate_BPM": np.random.randint(55, 100, n),
        "Daily_Steps": np.random.randint(1000, 15000, n),
        "Caffeine_Intake_mg": np.random.randint(0, 400, n),
        "Screen_Time_Before_Bed_Mins": np.random.randint(0, 120, n),
        "Alcohol_Units_Per_Week": np.random.randint(0, 20, n),
        "Room_Temperature_C": np.round(np.random.normal(20, 2, n), 1),
        "Noise_Level_dB": np.random.randint(20, 80, n),
        "Work_Hours_Per_Day": np.random.randint(4, 14, n),
        "Exercise_Type": np.random.choice(exercise_types, n),
        "Mental_Health_Score": np.random.randint(1, 11, n),
        "Awakenings_Per_Night": np.random.randint(0, 8, n),
        "Nap_Duration_Mins": np.random.randint(0, 90, n),
        "Is_Weekend": np.random.choice([0, 1], n),
        "Sleep_Disorder": np.random.choice(disorders, n, p=[0.6, 0.25, 0.15]),
        "Quality_of_Sleep": np.random.randint(1, 11, n),
    })

df["Sleep_Disorder"] = df["Sleep_Disorder"].fillna("None")

# Feature engineering (matches notebook)
df["Sleep_Deficit"] = np.maximum(0, 8 - df["Sleep_Duration_Hours"])
df["Activity_Stress_Ratio"] = (
    df["Physical_Activity_Mins"] / (df["Stress_Level"] + 1)
).round(2)

print(f"  Dataset shape: {df.shape}")

# ── 2. Feature / Target Setup ────────────────────────────────────────────────
FEATURES = [
    "Age", "Gender", "Occupation", "BMI_Category", "Smoking_Status",
    "Sleep_Duration_Hours", "Physical_Activity_Mins", "Stress_Level",
    "Heart_Rate_BPM", "Daily_Steps", "Caffeine_Intake_mg",
    "Screen_Time_Before_Bed_Mins", "Alcohol_Units_Per_Week",
    "Room_Temperature_C", "Noise_Level_dB", "Work_Hours_Per_Day",
    "Exercise_Type", "Mental_Health_Score", "Awakenings_Per_Night",
    "Nap_Duration_Mins", "Is_Weekend", "Sleep_Deficit", "Activity_Stress_Ratio"
]

TARGET_CLASS = "Sleep_Disorder"
TARGET_REG   = "Quality_of_Sleep"

X = df[FEATURES]
NUM = X.select_dtypes(include=np.number).columns.tolist()
CAT = [c for c in FEATURES if c not in NUM]

print(f"  Numeric features: {NUM}")
print(f"  Categorical features: {CAT}")

# ── 3. Preprocessor ──────────────────────────────────────────────────────────
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, NUM),
    ("cat", cat_pipe, CAT)
])

# ── 4. Classification: Sleep Disorder ────────────────────────────────────────
print("\n-- Classification: Sleep Disorder --")
le = LabelEncoder()
y_class = le.fit_transform(df[TARGET_CLASS].dropna())
X_cls   = df.loc[df[TARGET_CLASS].notna(), FEATURES]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_cls, y_class, test_size=0.2, random_state=42, stratify=y_class
)

clf_pipe = Pipeline([
    ("prep",  preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, max_depth=12,
                                     random_state=42, n_jobs=-1))
])
clf_pipe.fit(X_tr, y_tr)
preds_c = clf_pipe.predict(X_te)
acc  = accuracy_score(y_te, preds_c)
f1   = f1_score(y_te, preds_c, average="weighted")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1 Score : {f1:.4f}")

joblib.dump(clf_pipe, "classifier.pkl")
joblib.dump(le,       "label_encoder.pkl")
print("  Saved: classifier.pkl, label_encoder.pkl")

# ── 5. Regression: Quality of Sleep ──────────────────────────────────────────
print("\n-- Regression: Quality of Sleep --")
y_reg = df[TARGET_REG]
X_reg = df[FEATURES]

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_pipe = Pipeline([
    ("prep",  preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=12,
                                    random_state=42, n_jobs=-1))
])
reg_pipe.fit(X_tr_r, y_tr_r)
preds_r = reg_pipe.predict(X_te_r)
mae  = mean_absolute_error(y_te_r, preds_r)
rmse = np.sqrt(mean_squared_error(y_te_r, preds_r))
r2   = r2_score(y_te_r, preds_r)
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")

joblib.dump(reg_pipe, "regressor.pkl")
print("  Saved: regressor.pkl")

# ── 6. Feature Importance ────────────────────────────────────────────────────
rf_clf = clf_pipe.named_steps["model"]
feat_names = (
    NUM +
    list(clf_pipe.named_steps["prep"]
         .named_transformers_["cat"]
         .named_steps["onehot"]
         .get_feature_names_out(CAT))
)
importances = dict(zip(
    feat_names,
    rf_clf.feature_importances_.round(4).tolist()
))
top_features = sorted(importances, key=importances.get, reverse=True)[:10]

# ── 7. Metadata ──────────────────────────────────────────────────────────────
metadata = {
    "features": FEATURES,
    "numeric_features": NUM,
    "categorical_features": CAT,
    "disorder_classes": le.classes_.tolist(),
    "classifier_metrics": {"accuracy": round(acc, 4), "f1_score": round(f1, 4)},
    "regressor_metrics":  {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)},
    "top_features": top_features,
    "feature_importances": {k: importances[k] for k in top_features},
    "categorical_options": {
        "Gender":        sorted(df["Gender"].dropna().unique().tolist()),
        "Occupation":    sorted(df["Occupation"].dropna().unique().tolist()),
        "BMI_Category":  sorted(df["BMI_Category"].dropna().unique().tolist()),
        "Smoking_Status":sorted(df["Smoking_Status"].dropna().unique().tolist()),
        "Exercise_Type": sorted(df["Exercise_Type"].dropna().unique().tolist()),
    }
}

with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n[OK] Training complete -- all artifacts saved to models/")
print(json.dumps({
    "Classifier Accuracy": acc,
    "Classifier F1": f1,
    "Regressor R²": r2
}, indent=2))