from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import gspread
from oauth2client.service_account import ServiceAccountCredentials

app = FastAPI()

# Load model and binarizers
model = joblib.load("flareup_predictor_model.pkl")
mlb_meals = joblib.load("mlb_meals.pkl")
mlb_symptoms = joblib.load("mlb_symptoms.pkl")

# Google Sheets setup
SHEET_ID = "1J4Gh8VUnWeEyJGFT6vWYSCsOrlhTqRD8p82O6HXrhzU"  # 👈 Replace with your actual sheet ID

def log_to_google_sheets(data_dict):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID).sheet1
    headers = sheet.row_values(1)
    row = [data_dict.get(col, "") for col in headers]
    sheet.append_row(row)

# Define expected input format
class FlareInput(BaseModel):
    Age: int
    Gender: str  # "male" or "female"
    PainRating: int
    StressLevel: int
    TakeUlcerMed: str  # "Yes", "No", "Not sure"
    AteTriggers: str
    SkippedMeal: str
    AteLate: str
    TookNSAID: str
    CancerDiag: str
    FamilyHistory: str
    HpyloriUlcer: str
    Duration: str  # "<30 mins", "30 mins–2 hrs", ">2 hrs"
    Meals: list[str]
    Symptoms: list[str]

@app.post("/predict")
def predict_flare(data: FlareInput):
    # Convert input to dict
    input_data = data.dict()

    # Binary mapping
    binary_map = {'Yes': 1, 'No': 0, 'Not Sure': np.nan}
    for field in ['TakeUlcerMed', 'AteTriggers', 'SkippedMeal', 'AteLate', 'TookNSAID', 'CancerDiag', 'FamilyHistory', 'HpyloriUlcer']:
        input_data[field] = binary_map.get(input_data[field].title(), np.nan)

    # Gender encoding
    input_data['Gender'] = 1 if input_data['Gender'].strip().lower() == 'female' else 0

    # Duration mapping
    duration_map = {'<30 mins': 0, '30 mins–2 hrs': 1, '>2 hrs': 2}
    input_data['Duration'] = duration_map.get(input_data['Duration'], np.nan)

    # One-hot encode Meals and Symptoms
    meals_array = mlb_meals.transform([input_data['Meals']])
    symptoms_array = mlb_symptoms.transform([input_data['Symptoms']])
    meals_df = pd.DataFrame(meals_array, columns=[f"Meal_{m}" for m in mlb_meals.classes_])
    symptoms_df = pd.DataFrame(symptoms_array, columns=[f"Symptom_{s}" for s in mlb_symptoms.classes_])

    # Drop original multi-label fields
    base_df = pd.DataFrame([input_data])
    base_df = base_df.drop(columns=["Meals", "Symptoms"])

    # Combine all features
    model_input = pd.concat([base_df.reset_index(drop=True), meals_df, symptoms_df], axis=1)

    # Handle missing columns if any
    for col in model.feature_names_in_:
        if col not in model_input.columns:
            model_input[col] = 0

    # Reorder to match model
    model_input = model_input[model.feature_names_in_]

    # Predict
    prediction = model.predict(model_input)[0]

    # Log to Google Sheets
    input_data['Prediction'] = int(prediction)
    input_data['Meals'] = ';'.join(data.Meals)
    input_data['Symptoms'] = ';'.join(data.Symptoms)
    log_to_google_sheets(input_data)

    return {"flare_prediction": int(prediction)}




