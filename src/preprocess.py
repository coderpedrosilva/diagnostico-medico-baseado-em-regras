import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Colunas onde zero é clinicamente impossível e representa dado ausente
_CLINICAL_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base_dir, "data", "diabetes.csv"))

    # Substitui zeros por NaN nas colunas clínicas e imputa com a mediana
    df[_CLINICAL_COLS] = df[_CLINICAL_COLS].replace(0, np.nan)
    df[_CLINICAL_COLS] = df[_CLINICAL_COLS].fillna(df[_CLINICAL_COLS].median())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    return train_test_split(X, y, test_size=0.3, random_state=42)
