# Sleep Quality Prediction

An end-to-end machine learning system that predicts sleep disorder classification (None / Insomnia / Sleep Apnea) and sleep quality score regression from 23 lifestyle, physiological, and behavioral features — served via a Flask REST API with an interactive analytics dashboard and real-time predictor UI.

---

##  Features (23 input variables):
 
| Category | Features |
|---|---|
| Demographics | Age, Gender, Occupation, BMI Category, Smoking Status |
| Sleep | Sleep Duration (hrs), Awakenings Per Night, Nap Duration (mins), Is Weekend |
| Activity | Physical Activity (mins), Daily Steps, Exercise Type |
| Health | Heart Rate (BPM), Mental Health Score, Stress Level |
| Lifestyle | Caffeine Intake (mg), Alcohol Units/Week, Screen Time Before Bed (mins), Work Hours/Day |
| Environment | Room Temperature (°C), Noise Level (dB) |
| Engineered | Sleep Deficit (`max(0, 8 − Sleep_Duration)`), Activity-Stress Ratio (`Activity_Mins / (Stress + 1)`) |
 
**Targets:**
- `Sleep_Disorder` — 3-class classification: None / Insomnia / Sleep Apnea
- `Quality_of_Sleep` — regression, score 1–10

---

## Steps to Run
 
### 1. Clone the repository
 
```bash
git clone https://github.com/vuchau0802/Sleep-Quality-Prediction.git
cd Sleep-Quality-Prediction
```
 
### 2. Create and activate a virtual environment
 
```bash
# Windows
python -m venv venv
venv\Scripts\activate
 
# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt   
```

### 4. Train the models
 
```bash
python train_model.py
```


### 5. Start the Flask API
 
```bash
python app.py
```

