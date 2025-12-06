from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from typing import Optional
import json
from inference import process_hand_image
from database import get_db_connection, execute_query, execute_insert

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
async def detect_hand(
    file: UploadFile = File(...),
    campaign_id: Optional[str] = Form(None)
):
    try:
        # Log the received campaign_id
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
        
        # Get referentiels for this campaign and match sizes
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
                            
                            # Extract hand measurement criteria from JSON
                            longueur_main = value_data.get('longueur_main', {})  # hand length
                            longueur_paume = value_data.get('longueur_paume', {})  # palm length
                            tour_main = value_data.get('tour_main', {})  # hand perimeter
                            tour_poignet = value_data.get('tour_poignet', {})  # wrist perimeter
                            
                            # Get all available sizes
                            all_sizes = set()
                            for criterion in [longueur_main, longueur_paume, tour_main, tour_poignet]:
                                all_sizes.update(criterion.keys())
                            
                            # Calculate score for each size based on all criteria
                            size_scores = {}
                            
                            for size in all_sizes:
                                total_diff = 0
                                criteria_count = 0
                                
                                # Check tour_main (hand perimeter) - most important
                                if size in tour_main and tour_main[size] and tour_main[size] != "":
                                    try:
                                        ref_value = float(tour_main[size])
                                        diff = abs(ref_value - measurements["tour_main"])
                                        total_diff += diff * 2  # Weight this criterion more
                                        criteria_count += 2
                                    except (ValueError, TypeError):
                                        pass
                                
                                # Check longueur_paume (palm length)
                                if size in longueur_paume and longueur_paume[size] and longueur_paume[size] != "":
                                    try:
                                        ref_value = float(longueur_paume[size])
                                        diff = abs(ref_value - measurements["longueur_paume"])
                                        total_diff += diff
                                        criteria_count += 1
                                    except (ValueError, TypeError):
                                        pass
                                
                                # Check longueur_main (hand length)
                                if size in longueur_main and longueur_main[size] and longueur_main[size] != "":
                                    try:
                                        ref_value = float(longueur_main[size])
                                        diff = abs(ref_value - measurements["longueur_main"])
                                        total_diff += diff
                                        criteria_count += 1
                                    except (ValueError, TypeError):
                                        pass
                                
                                # Check tour_poignet (wrist perimeter)
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
                            
                            # Find the best matching size
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


@app.get("/test-db")
async def test_database():
    """Test database connection and show campaign_referentiel and referentiel table structures"""
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # Get database info
            cursor.execute("SELECT DATABASE(), VERSION()")
            db_info = cursor.fetchone()
            
            # Get campaign_referentiel table structure
            cursor.execute("DESCRIBE campaign_referentiel")
            campaign_referentiel_structure = cursor.fetchall()
            
            # Get referentiel table structure
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
        # Get referentiel IDs from campaign_referentiel table
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
