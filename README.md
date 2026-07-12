# Heart Disease Prediction API

An end-to-end Machine Learning deployment project that predicts the risk of heart disease based on clinical parameters. This project bridges biomedical engineering data analysis with production-ready software development using a high-performance REST API.

---

## Research & Methodology
The complete theoretical background, feature engineering process, and model evaluation metrics for this project are documented in a formal research paper.
* **Format:** IEEE Style
* **Location:** You can access the full PDF in the repository under `docs/Heart_Disease_Prediction_Paper.pdf`.

---

## Tech Stack & Libraries
* **Language:** Python
* **API Framework:** FastAPI (with Uvicorn ASGI server)
* **Data Validation:** Pydantic
* **Machine Learning:** Scikit-Learn 
* **Data Manipulation:** Pandas
* **Model Serialization:** Joblib

---

## Features & Data Structure
The underlying model utilizes **15 distinct clinical features** after undergoing proper One-Hot Encoding during the data preprocessing stage to handle categorical variables cleanly and avoid multi-collinearity:

* `Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`
* `Sex_M` (Binary encoded)
* `ChestPainType_ATA`, `ChestPainType_NAP`, `ChestPainType_TA`
* `RestingECG_Normal`, `RestingECG_ST`
* `ExerciseAngina_Y`
* `ST_Slope_Flat`, `ST_Slope_Up`

---

## Installation & Local Setup

To clone and run this API locally, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/HeartDiseasePrediction.git
cd HeartDiseasePrediction
```

### 2. Set up a Virtual Environment
```bash
python -m venv .venv
```
* **Activate on Windows:**
  ```bash
  .venv\Scripts\activate
  ```
* **Activate on macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI Server
```bash
uvicorn src.main:app --reload
```

---

## API Usage & Interactive Documentation

Once the server is running, FastAPI automatically generates interactive Swagger UI documentation. 

* **Interactive Interface:** Open your browser and navigate to `http://127.0.0.1:8000/docs`
* **Health Check:** `GET http://127.0.0.1:8000/`

### Example Request Body (`POST /predikcija`)
```json
{
  "Age": 50,
  "RestingBP": 140,
  "Cholesterol": 289,
  "FastingBS": 0,
  "MaxHR": 172,
  "Oldpeak": 0.0,
  "Sex_M": 1,
  "ChestPainType_ATA": 1,
  "ChestPainType_NAP": 0,
  "ChestPainType_TA": 0,
  "RestingECG_Normal": 1,
  "RestingECG_ST": 0,
  "ExerciseAngina_Y": 0,
  "ST_Slope_Flat": 0,
  "ST_Slope_Up": 1
}
```

### Example Response
```json
{
  "status": "Uspesno",
  "rizik_od_bolesti": 1,
  "verovatnoca_bolesti": 0.9999
}
```
