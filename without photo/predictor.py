# predictor.py
import os, joblib, pandas as pd

# Get the directory where this predictor.py file is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.getenv("WEEFIZZ_MODELS_BASE", os.path.join(SCRIPT_DIR, "models"))

REGRESSORS_PATH = {
    "handlength":        os.path.join(BASE_PATH, "handlength_regressor.pkl"),
    "palmlength":        os.path.join(BASE_PATH, "palmlength_regressor.pkl"),
    "handcircumference": os.path.join(BASE_PATH, "handcircumference_regressor.pkl"),
    "wristcircumference":os.path.join(BASE_PATH, "wristcircumference_regressor.pkl"),
}
CLASSIFIER_PATH    = os.path.join(BASE_PATH, "glove_classifier_from_predicted_measurements.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_PATH, "hand_size_encoder.pkl")


REGRESSORS    = {k: joblib.load(v) for k, v in REGRESSORS_PATH.items()}
CLASSIFIER    = joblib.load(CLASSIFIER_PATH)
LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH)

def predict_size_from_demo(age: float, sex: str, height_cm: float, weight_kg: float):
    """sex: 'M' or 'F'"""
    gender = 1 if sex.strip().upper() == "F" else 0  
    df_demo = pd.DataFrame([[age, gender, height_cm, weight_kg]],
                           columns=["Age", "Gender", "height_cm", "weight_kg"])

    # Stage 1: regress 4 measures (models output mm)
    preds_mm = {name: float(model.predict(df_demo)[0]) for name, model in REGRESSORS.items()}
    preds_cm = {k: round(v / 10.0, 1) for k, v in preds_mm.items()}  # pretty output

    # Stage 2: classify glove size from (mm) measurements
    df_for_clf = pd.DataFrame([preds_mm])
    y_pred_class = CLASSIFIER.predict(df_for_clf)[0]
    size_label = LABEL_ENCODER.inverse_transform([y_pred_class])[0]

    return {"measurements_cm": preds_cm, "size_label": str(size_label)}
