import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

class ModelPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_assets()

    def load_assets(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            print("Model aur Scaler load ho gaye hain!")
        except Exception as e:
            print(f"⚠️ Error: {e}. Check karo ki model folder me files hain ya nahi.")

    def predict(self, data) -> float:
        if not self.model:
            raise Exception("Model files missing")

        # 1. 7 inputs ka array banaya
        input_features = [[
            data.age,
            data.study_hours_per_day,
            data.deep_work_sessions,
            data.assignment_completion_rate,
            data.attendance_percentage,
            data.stress_level,
            data.motivation_level
        ]]

        # 2. Input ko scale kiya (scaler.pkl ka use karke)
        scaled_features = self.scaler.transform(input_features)

        # 3. Model se score predict karwaya
        prediction = self.model.predict(scaled_features)

        # Yeh ek array return karta hai jaise [85.432], toh hume first item chahiye
        return float(prediction[0])

predictor = ModelPredictor()