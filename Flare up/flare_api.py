from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import gspread
from oauth2client.service_account import ServiceAccountCredentials

app = FastAPI()

# Load model and encoders
model = joblib.load("flareup_predictor_model.pkl")
mlb_meals = joblib.load("mlb_meals.pkl")
mlb_symptoms = joblib.load("mlb_symptoms.pkl")

# Google Sheets setup
SHEET_ID = "1J4Gh8VUnWeEyJGFT6vWYSCsOrlhTqRD8p82O6HXrhzU"  

def log_to_google_sheets(data_dict):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID).sheet1
    headers = sheet.row_values(1)
    row = [data_dict.get(col, "") for col in headers]
    sheet.append_row(row)

# Define expected input
class FlareInput(BaseModel):
    Age: int
    Gender: str
    PainRating: int
    StressLevel: int
    TakeUlcerMed: str
    AteTriggers: str
    SkippedMeal: str
    AteLate: str
    TookNSAID: str
    CancerDiag: str
    FamilyHistory: str
    HpyloriUlcer: str
    Duration: str
    Meals: list[str]
    Symptoms: list[str]

@app.post("/predict")
def predict_flare(data: FlareInput):
    # Prepare original input for logging
    input_data = data.dict()

    # Keep a copy before transformation
    log_data = input_data.copy()

    # Binary encoding
    binary_map = {'Yes': 1, 'No': 0, 'Not Sure': np.nan}
    for field in ['TakeUlcerMed', 'AteTriggers', 'SkippedMeal', 'AteLate', 'TookNSAID', 'CancerDiag', 'FamilyHistory', 'HpyloriUlcer']:
        input_data[field] = binary_map.get(input_data[field].title(), np.nan)

    # Encode gender
    input_data['Gender'] = 1 if input_data['Gender'].strip().lower() == 'female' else 0

    # Encode duration
    duration_map = {'<30 mins': 0, '30 mins–2 hrs': 1, '>2 hrs': 2}
    input_data['Duration'] = duration_map.get(input_data['Duration'], np.nan)

    # Multi-label encoding
    meals_array = mlb_meals.transform([input_data['Meals']])
    symptoms_array = mlb_symptoms.transform([input_data['Symptoms']])
    meals_df = pd.DataFrame(meals_array, columns=[f"Meal_{m}" for m in mlb_meals.classes_])
    symptoms_df = pd.DataFrame(symptoms_array, columns=[f"Symptom_{s}" for s in mlb_symptoms.classes_])

    # Drop raw fields
    base_df = pd.DataFrame([input_data]).drop(columns=["Meals", "Symptoms"])

    # Final input
    model_input = pd.concat([base_df.reset_index(drop=True), meals_df, symptoms_df], axis=1)

    # Handle missing features
    for col in model.feature_names_in_:
        if col not in model_input.columns:
            model_input[col] = 0
    model_input = model_input[model.feature_names_in_]

    # Predict
    prediction = model.predict(model_input)[0]
    prediction_label = "Yes" if prediction == 1 else "No"

    # Prepare readable data for logging
    log_data['Meals'] = ';'.join(data.Meals)
    log_data['Symptoms'] = ';'.join(data.Symptoms)
    log_data['Prediction'] = prediction_label

    reverse_binary = {1: "Yes", 0: "No", np.nan: "Not Sure"}
    for field in ['TakeUlcerMed', 'AteTriggers', 'SkippedMeal', 'AteLate', 'TookNSAID', 'CancerDiag', 'FamilyHistory', 'HpyloriUlcer']:
        log_data[field] = reverse_binary.get(input_data[field], "Not Sure")

    log_data['Gender'] = "Female" if input_data['Gender'] == 1 else "Male"
    reverse_duration = {0: "<30 mins", 1: "30 mins–2 hrs", 2: ">2 hrs"}
    log_data['Duration'] = reverse_duration.get(input_data['Duration'], input_data['Duration'])

    # Save to sheet
    log_to_google_sheets(log_data)

    return {"flare_prediction": prediction_label}

