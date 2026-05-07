# Sleep Quality Prediction using Machine Learning

An end-to-end Machine Learning project for predicting sleep quality based on lifestyle, health, and behavioral factors.

This project applies data preprocessing, feature engineering, exploratory data analysis (EDA), and multiple machine learning models to analyze sleep patterns and predict sleep quality outcomes.

---

##  Key Features

### Dataset — 35 Features × 110,000 Rows

The dataset covers eight groups of features across 110,000 synthetic patient records.

**Demographics** — `Age`, `Gender`, `Occupation` (21 job types: Doctor, Engineer, Teacher, Nurse, etc.), `Is_Weekend`

**Body Metrics** — `BMI_Value`, `BMI_Category` (Underweight / Normal / Overweight / Obese), `Smoking_Status` (Never / Former / Current)

**Sleep Core** — `Sleep_Duration_Hours`, `Bedtime`, `Wake_Up_Time`, `Quality_of_Sleep` (score 1–10), `Sleep_Efficiency_Pct`

**Sleep Architecture** — `REM_Sleep_Pct`, `Deep_Sleep_Pct`, `Light_Sleep_Pct`, `Awakenings_Per_Night`, `Nap_Duration_Mins`

**Lifestyle** — `Physical_Activity_Mins`, `Exercise_Type` (Walking / Running / Gym / Yoga / Swimming / Cycling / Mixed / None), `Stress_Level` (1–10), `Mental_Health_Score` (0–100), `Work_Hours_Per_Day`

**Vitals** — `Heart_Rate_BPM`, `Blood_Pressure`, `Systolic_BP`, `Diastolic_BP`, `Daily_Steps`

**Environment** — `Caffeine_Intake_mg`, `Screen_Time_Before_Bed_Mins`, `Alcohol_Units_Per_Week`, `Room_Temperature_C`, `Noise_Level_dB`

**Derived Features** — `Sleep_Deficit` (hours below the recommended 8 h), `Activity_Stress_Ratio` (physical activity ÷ stress level)

**Target — Classification:** `Sleep_Disorder` with three classes — None (82.4 %), Sleep Apnea (13.4 %), Insomnia (4.2 %)

**Target — Regression:** `Quality_of_Sleep` — integer score 1 to 10

---

##  Multiple ML models

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
