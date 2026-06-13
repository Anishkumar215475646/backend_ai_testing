from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.model_loader import predictor

app = FastAPI(title="Student Exam Score Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Naye model ke 7 inputs
class StudentData(BaseModel):
    age: int
    study_hours_per_day: float
    deep_work_sessions: int
    assignment_completion_rate: float
    attendance_percentage: float
    stress_level: float
    motivation_level: float

@app.get("/")
def home():
    return {"message": "Exam Score Predictor is running!"}

@app.post("/predict")
def predict_score(data: StudentData):
    try:
        # Result ab float me aayega
        score = predictor.predict(data)
        return {
            "status": "success",
            "predicted_score": round(score, 2) # Score ko 2 decimal places tak round off kiya
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))