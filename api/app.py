import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.preprocess import load_data
from src.majority import MajorityClassifier
from src.oner import OneR
from src.prism import Prism

# ======================================
# Inicialização da API
# ======================================
app = FastAPI(title="Medical Rule-Based Diagnosis")

# ======================================
# Treinar modelos uma única vez na inicialização
# ======================================
X_train, X_test, y_train, y_test = load_data()

majority = MajorityClassifier()
majority.fit(y_train)

oner = OneR()
oner.fit(X_train, y_train)

prism = Prism()
prism.fit(X_train, y_train)

_MODELS = {"majority": majority, "oner": oner, "prism": prism}

# ======================================
# Medianas do dataset (após imputação) usadas como padrão
# ======================================
class Patient(BaseModel):
    Pregnancies: float = Field(default=3.0, ge=0)
    Glucose: float = Field(default=117.0, gt=0)
    BloodPressure: float = Field(default=72.0, gt=0)
    SkinThickness: float = Field(default=29.0, gt=0)
    Insulin: float = Field(default=125.0, gt=0)
    BMI: float = Field(default=32.0, gt=0)
    DiabetesPedigreeFunction: float = Field(default=0.37, gt=0)
    Age: float = Field(default=29.0, gt=0)

# ======================================
# Endpoint de predição
# ======================================
@app.post("/predict/{model}")
def predict(model: str, p: Patient):
    if model not in _MODELS:
        raise HTTPException(status_code=404, detail=f"Modelo '{model}' não encontrado. Use: {list(_MODELS.keys())}")

    X = pd.DataFrame([[
        p.Pregnancies, p.Glucose, p.BloodPressure, p.SkinThickness,
        p.Insulin, p.BMI, p.DiabetesPedigreeFunction, p.Age
    ]], columns=X_train.columns)

    clf = _MODELS[model]
    pred = int(clf.predict(X)[0])

    if model == "majority":
        rule = "Classe majoritária da base de treino"
    elif model == "oner":
        rule = f"Regra única baseada em '{oner.best_feature}'"
    else:
        rule = f"Conjunto de {len(prism.rules)} regras PRISM"

    return {"prediction": pred, "rule": rule}

# ======================================
# Frontend (UI)
# ======================================
STATIC_DIR = os.path.join(BASE_DIR, "api", "static")
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="static")

@app.get("/")
def root():
    return RedirectResponse("/ui")
