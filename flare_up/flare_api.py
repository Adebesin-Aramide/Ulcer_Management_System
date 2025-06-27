from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load model and encoders
model = joblib.load("flareup_model.pkl")
mlb_meals = joblib.load("meals.pkl")
mlb_symptoms = joblib.load("symptoms.pkl")

class InputData(BaseModel):
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

@app.post("/")
def predict(data: InputData):
    input_dict = data.dict()

    binary_map = {'Yes': 1, 'No': 0, 'Not Sure': np.nan}
    for field in ['TakeUlcerMed', 'AteTriggers', 'SkippedMeal', 'AteLate',
                  'TookNSAID', 'CancerDiag', 'FamilyHistory', 'HpyloriUlcer']:
        input_dict[field] = binary_map.get(input_dict[field].title(), np.nan)

    input_dict['Gender'] = 1 if input_dict['Gender'].strip().lower() == 'female' else 0

    duration_map = {'<30 mins': 0, '30 mins–2 hrs': 1, '>2 hrs': 2}
    input_dict['Duration'] = duration_map.get(input_dict['Duration'], np.nan)

    meals_df = pd.DataFrame(mlb_meals.transform([input_dict['Meals']]),
                            columns=[f"Meal_{m}" for m in mlb_meals.classes_])
    symptoms_df = pd.DataFrame(mlb_symptoms.transform([input_dict['Symptoms']]),
                               columns=[f"Symptom_{s}" for s in mlb_symptoms.classes_])

    base_df = pd.DataFrame([input_dict]).drop(columns=["Meals", "Symptoms"])
    model_input = pd.concat([base_df, meals_df, symptoms_df], axis=1)

    for col in model.feature_names_in_:
        if col not in model_input.columns:
            model_input[col] = 0

    model_input = model_input[model.feature_names_in_]

    prediction = model.predict(model_input)[0]
    return {"flare_prediction": "Yes" if prediction == 1 else "No"}

