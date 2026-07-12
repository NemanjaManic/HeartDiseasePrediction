import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

putanja_do_modela = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
model = joblib.load(putanja_do_modela)

class Pacijent(BaseModel):
    Age: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    MaxHR: int
    Oldpeak: float
    Sex_M: int  # 1 ako je musko, 0 ako je zensko
    ChestPainType_ATA: int  # 1 ako je ATA, 0 u suprotnom
    ChestPainType_NAP: int  # 1 ako je NAP, 0 u suprotnom
    ChestPainType_TA: int  # 1 ako je TA, 0 u suprotnom
    RestingECG_Normal: int  # 1 ako je Normal, 0 u suprotnom
    RestingECG_ST: int  # 1 ako je ST, 0 u suprotnom
    ExerciseAngina_Y: int  # 1 ako je DA (Yes), 0 ako je NE (No)
    ST_Slope_Flat: int  # 1 ako je Flat, 0 u suprotnom
    ST_Slope_Up: int  # 1 ako je Up, 0 u suprotnom


@app.get("/")
def home():
    return {"poruka": "API sa 15 obelezja je spreman!"}

@app.post("/predikcija")
def napravi_predikciju(podaci: Pacijent):
    # model_dump() pretvara podatke iz klase u obican Python recnik
    ulazni_podaci = pd.DataFrame([podaci.model_dump()])

    # Izvršavanje predikcije
    predikcija = int(model.predict(ulazni_podaci)[0])
    verovatnoca = float(model.predict_proba(ulazni_podaci)[0][1])

    return {
        "status": "Uspesno",
        "rizik_od_bolesti": predikcija,
        "verovatnoca_bolesti": round(verovatnoca, 4)
    }