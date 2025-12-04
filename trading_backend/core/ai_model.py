# core/ai_model.py
import os
import joblib
import numpy as np

# مكان الموديل
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")

# تحميل الموديل مرة واحدة
print(f"Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print(f"Model loaded successfully: {model}")

def get_decision(features: list):
    """
    بياخد لستة features ويرجع التوقع.
    """
    try:
        # تأكد إن كلها float
        features = [float(x) for x in features]

        X = np.array([features])
        print("Input features to model:", X)

        prediction = model.predict(X)[0]
        print("Prediction from model:", prediction)

        return int(prediction)

    except Exception as e:
        print("Error in get_decision:", e)
        return {"error": str(e)}
