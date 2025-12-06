from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import shutil
import os
import sys
from typing import Optional
import json

# Add subfolders to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "with photo"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "without photo"))

# Import from "with photo" folder
from inference import process_hand_image
from database import get_db_connection, execute_query, execute_insert

# Import from "without photo" folder
from predictor import predict_size_from_demo

app = FastAPI(title="WeeFizz Hand Measurement API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
UPLOAD_FOLDER = "with photo/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "with photo/models/best.pt"
app.mount("/output", StaticFiles(directory="with photo/output"), name="output")

# ============ WITH PHOTO ENDPOINTS ============

@app.post("/detect/")
async def detect_hand(
    file: UploadFile = File(...),
    campaign_id: Optional[str] = Form(None)
):
    try:
        if campaign_id:
            print(f"Received campaign_id: {campaign_id}")
        
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = process_hand_image(image_path, MODEL_PATH)
        measurements = {
            "longueur_main": result["longueur_main"],
            "longueur_paume": result["longueur_paume"],
            "tour_main": result["tour_main"],
            "tour_poignet": result["tour_poignet"]
        }
        annotated_url = f"/output/{result['annotated_image_filename']}"

        recommended_sizes = []
        
        if campaign_id:
            try:
                query = """
                    SELECT r.id, r.name, r.value, r.image_hands
                    FROM campaign_referentiel cr
                    JOIN referentiel r ON cr.referentiel_id = r.id
                    WHERE cr.campaign_id = %s AND r.actif = 1
                """
                referentiels = execute_query(query, (campaign_id,))
                
                for ref in referentiels:
                    if ref['value']:
                        try:
                            value_data = json.loads(ref['value'])
                            
                            longueur_main = value_data.get('longueur_main', {})
                            longueur_paume = value_data.get('longueur_paume', {})
                            tour_main = value_data.get('tour_main', {})
                            tour_poignet = value_data.get('tour_poignet', {})
                            
                            all_sizes = set()
                            for criterion in [longueur_main, longueur_paume, tour_main, tour_poignet]:
                                all_sizes.update(criterion.keys())
                            
                            size_scores = {}
                            
                            for size in all_sizes:
                                total_diff = 0
                                criteria_count = 0
                                
                                if size in tour_main and tour_main[size] and tour_main[size] != "":
                                    try:
                                        ref_value = float(tour_main[size])
                                        diff = abs(ref_value - measurements["tour_main"])
                                        total_diff += diff * 2
                                        criteria_count += 2
                                    except (ValueError, TypeError):
                                        pass
                                
                                if size in longueur_paume and longueur_paume[size] and longueur_paume[size] != "":
                                    try:
                                        ref_value = float(longueur_paume[size])
                                        diff = abs(ref_value - measurements["longueur_paume"])
                                        total_diff += diff
                                        criteria_count += 1
                                    except (ValueError, TypeError):
                                        pass
                                
                                if size in longueur_main and longueur_main[size] and longueur_main[size] != "":
                                    try:
                                        ref_value = float(longueur_main[size])
                                        diff = abs(ref_value - measurements["longueur_main"])
                                        total_diff += diff
                                        criteria_count += 1
                                    except (ValueError, TypeError):
                                        pass
                                
                                if size in tour_poignet and tour_poignet[size] and tour_poignet[size] != "":
                                    try:
                                        ref_value = float(tour_poignet[size])
                                        diff = abs(ref_value - measurements["tour_poignet"])
                                        total_diff += diff
                                        criteria_count += 1
                                    except (ValueError, TypeError):
                                        pass
                                
                                if criteria_count > 0:
                                    avg_diff = total_diff / criteria_count
                                    size_scores[size] = avg_diff
                            
                            if size_scores:
                                best_size = min(size_scores, key=size_scores.get)
                                
                                recommended_sizes.append({
                                    "referentiel_id": ref['id'],
                                    "referentiel_name": ref['name'],
                                    "recommended_size": best_size,
                                    "criteria_match": {
                                        "tour_main": tour_main.get(best_size),
                                        "longueur_paume": longueur_paume.get(best_size),
                                        "longueur_main": longueur_main.get(best_size),
                                        "tour_poignet": tour_poignet.get(best_size)
                                    },
                                    "average_difference_mm": round(size_scores[best_size], 2),
                                    "image": ref['image_hands']
                                })
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON for referentiel {ref['id']}")
                            continue
                
            except Exception as db_error:
                print(f"Database error: {db_error}")

        return {
            "measurements": measurements,
            "campaign_id": campaign_id,
            "recommended_sizes": recommended_sizes
        }

    except Exception as e:
        print(f"Error in /detect/ endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})


# ============ WITHOUT PHOTO ENDPOINTS ============

class Input(BaseModel):
    age: float = Field(..., ge=14, le=90)
    sex: str = Field(..., pattern="^(M|F|m|f)$")
    height_cm: float = Field(..., gt=100, lt=230)
    weight_kg: float = Field(..., gt=30, lt=220)
    campaign_id: Optional[str] = None

class Output(BaseModel):
    measurements: dict
    campaign_id: Optional[str] = None
    recommended_sizes: list = []

@app.post("/predict-glove-size", response_model=Output)
def predict(inp: Input):
    try:
        print(f"=== PREDICT GLOVE SIZE REQUEST ===")
        print(f"Received campaign_id: {inp.campaign_id}")
        
        # Get predicted measurements from ML model
        res = predict_size_from_demo(inp.age, inp.sex, inp.height_cm, inp.weight_kg)
        
        # Convert measurements from cm to mm for database matching
        # Note: ML model predicts hand circumference, but database stores hand width
        # Convert circumference to width: width ≈ circumference / 2 (hand is flat, not circular)
        hand_circumference_mm = res["measurements_cm"]["handcircumference"] * 10
        hand_width_mm = hand_circumference_mm / 2
        
        measurements = {
            "longueur_main": res["measurements_cm"]["handlength"] * 10,
            "longueur_paume": res["measurements_cm"]["palmlength"] * 10,
            "tour_main": hand_width_mm,  # Convert circumference to width
            "tour_poignet": res["measurements_cm"]["wristcircumference"] * 10
        }
        
        print(f"Predicted measurements (mm): {measurements}")
        print(f"Note: Converted hand circumference {hand_circumference_mm}mm to width {hand_width_mm}mm")
        
        recommended_sizes = []
        
        # If campaign_id provided, match against database referentiels
        if inp.campaign_id:
            print(f"Querying database for campaign_id: {inp.campaign_id}")
            try:
                query = """
                    SELECT r.id, r.name, r.value, r.image_hands
                    FROM campaign_referentiel cr
                    JOIN referentiel r ON cr.referentiel_id = r.id
                    WHERE cr.campaign_id = %s AND r.actif = 1
                """
                referentiels = execute_query(query, (inp.campaign_id,))
                
                print(f"Found {len(referentiels)} referentiels for campaign {inp.campaign_id}")
                
                for ref in referentiels:
                    if ref['value']:
                        try:
                            value_data = json.loads(ref['value'])
                            
                            longueur_main = value_data.get('longueur_main', {})
                            longueur_paume = value_data.get('longueur_paume', {})
                            tour_main = value_data.get('tour_main', {})
                            tour_poignet = value_data.get('tour_poignet', {})
                            
                            all_sizes = set()
                            for criterion in [longueur_main, longueur_paume, tour_main, tour_poignet]:
                                all_sizes.update(criterion.keys())
                            
                            size_scores = {}
                            
                            for size in all_sizes:
                                total_diff = 0
                                criteria_count = 0
                                
                                if size in tour_main and tour_main[size] and tour_main[size] != "":
                                    try:
                                        ref_value = float(tour_main[size])
                                        diff = abs(ref_value - measurements["tour_main"])
                                        total_diff += diff * 2
                                        criteria_count += 2
                                    except (ValueError, TypeError):
                                        pass
                                
                                if size in longueur_paume and longueur_paume[size] and longueur_paume[size] != "":
                                    try:
                                        ref_value = float(longueur_paume[size])
                                        diff = abs(ref_value - measurements["longueur_paume"])
                                        total_diff += diff
                                        criteria_count += 1
                                    except (ValueError, TypeError):
                                        pass
                                
                                if size in longueur_main and longueur_main[size] and longueur_main[size] != "":
                                    try:
                                        ref_value = float(longueur_main[size])
                                        diff = abs(ref_value - measurements["longueur_main"])
                                        total_diff += diff
                                        criteria_count += 1
                                    except (ValueError, TypeError):
                                        pass
                                
                                if size in tour_poignet and tour_poignet[size] and tour_poignet[size] != "":
                                    try:
                                        ref_value = float(tour_poignet[size])
                                        diff = abs(ref_value - measurements["tour_poignet"])
                                        total_diff += diff
                                        criteria_count += 1
                                    except (ValueError, TypeError):
                                        pass
                                
                                if criteria_count > 0:
                                    avg_diff = total_diff / criteria_count
                                    size_scores[size] = avg_diff
                            
                            if size_scores:
                                best_size = min(size_scores, key=size_scores.get)
                                
                                recommended_sizes.append({
                                    "referentiel_id": ref['id'],
                                    "referentiel_name": ref['name'],
                                    "recommended_size": best_size,
                                    "criteria_match": {
                                        "tour_main": tour_main.get(best_size),
                                        "longueur_paume": longueur_paume.get(best_size),
                                        "longueur_main": longueur_main.get(best_size),
                                        "tour_poignet": tour_poignet.get(best_size)
                                    },
                                    "average_difference_mm": round(size_scores[best_size], 2),
                                    "image": ref['image_hands']
                                })
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON for referentiel {ref['id']}")
                            continue
                
            except Exception as db_error:
                print(f"Database error: {db_error}")
        else:
            print("No campaign_id provided - skipping database matching")
        
        # Return ML model prediction as recommended_size, but also include database matches
        print(f"Returning {len(recommended_sizes)} recommended sizes")
        return {
            "measurements": measurements,
            "campaign_id": inp.campaign_id,
            "recommended_sizes": recommended_sizes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ DATABASE ENDPOINTS ============

@app.get("/test-db")
async def test_database():
    """Test database connection and show campaign_referentiel and referentiel table structures"""
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT DATABASE(), VERSION()")
            db_info = cursor.fetchone()
            
            cursor.execute("DESCRIBE campaign_referentiel")
            campaign_referentiel_structure = cursor.fetchall()
            
            cursor.execute("DESCRIBE referentiel")
            referentiel_structure = cursor.fetchall()
            
        connection.close()
        return {
            "status": "success",
            "database": db_info['DATABASE()'],
            "version": db_info['VERSION()'],
            "campaign_referentiel_structure": campaign_referentiel_structure,
            "referentiel_structure": referentiel_structure
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.get("/campaign/{campaign_id}/referentiels")
async def get_campaign_referentiels(campaign_id: str):
    """Get referentiels related to a campaign"""
    try:
        query = """
            SELECT cr.*, r.* 
            FROM campaign_referentiel cr
            JOIN referentiel r ON cr.referentiel_id = r.id
            WHERE cr.campaign_id = %s
        """
        results = execute_query(query, (campaign_id,))
        
        return {
            "status": "success",
            "campaign_id": campaign_id,
            "referentiels": results
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


# ============ HEALTH CHECK ============

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "message": "WeeFizz Hand Measurement API",
        "endpoints": {
            "with_photo": "/detect/",
            "without_photo": "/predict-glove-size",
            "database": ["/test-db", "/campaign/{campaign_id}/referentiels"],
            "health": "/healthz"
        }
    }
