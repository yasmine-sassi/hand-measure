from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from inference import process_hand_image, recommend_glove_size

app = FastAPI()

# CORS pour autoriser l'appli frontend à accéder à l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en prod
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "models/best.pt"
from fastapi.staticfiles import StaticFiles

app.mount("/output", StaticFiles(directory="output"), name="output")

import base64

@app.post("/detect/")
async def detect_hand(file: UploadFile = File(...)):
    try:
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = process_hand_image(image_path, MODEL_PATH)
        measurements = {
            "palm_width_mm": result["palm_width_mm"],
            "index_length_mm": result["index_length_mm"],
            "hand_perimeter_mm": result["hand_perimeter_mm"]
        }
        glove_size = recommend_glove_size(measurements["palm_width_mm"])
        annotated_url = f"/output/{result['annotated_image_filename']}"

        return {
            "measurements": measurements,
            "glove_size": glove_size,
        }



    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

