import cv2
import mediapipe as mp
import numpy as np
import os
from math import sqrt
from ultralytics import YOLO

# === Initialisation ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Dictionnaire des diamètres des pièces en mm
euro_coins = {
    "1 cent": 16.25,
    "2 cent": 18.75,
    "5 cent": 21.25,
    "10 cent": 19.75,
    "20 cent": 22.25,
    "50 decorr": 24.25,
    "1 euro": 23.25,
    "2 euro": 25.75
}

def calculate_distance(p1, p2):
    return sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def pixels_to_mm(distance_px, ref_px, ref_mm):
    if ref_px == 0:
        raise ValueError("Référence en pixels = 0")
    return (distance_px * ref_mm) / ref_px

def detect_coin_with_yolo(image, model):
    results = model(image)
    if results[0].boxes and len(results[0].boxes) > 0:
        best_box = max(results[0].boxes, key=lambda x: x.conf)
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cls = int(best_box.cls[0])
        class_name = results[0].names[cls]
        radius = (x2 - x1) // 2
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return center_x, center_y, radius, class_name
    return None, None, None, None

def process_hand_image(image_path, model_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image non chargée")

    model = YOLO(model_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # === Détection de la pièce ===
    x, y, r, class_name = detect_coin_with_yolo(image_rgb, model)
    if None in (x, y, r):
        raise ValueError("Pièce non détectée")

    # Dessin de la pièce détectée
    cv2.circle(image, (x, y), r, (0, 255, 255), 2)
    cv2.putText(image, class_name, (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    coin_diameter_px = r * 2
    reference_mm = euro_coins.get(class_name, 27)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            raise ValueError("Main non détectée")

        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        lm = [(l.x * image.shape[1], l.y * image.shape[0]) for l in hand_landmarks.landmark]
        
        # longueur_main (hand length): wrist to middle finger tip
        hand_length_px = calculate_distance(lm[0], lm[12])
        
        # longueur_paume (palm length): wrist to base of middle finger
        palm_length_px = calculate_distance(lm[0], lm[9])
        
        # tour_main (hand perimeter): around the widest part of the hand
        # Based on referentiel values (91-115mm), this appears to be hand width across knuckles
        hand_width_px = calculate_distance(lm[5], lm[17])  # index base to pinky base
        hand_perimeter_px = hand_width_px  # Direct measurement, not circumference
        
        # tour_poignet (wrist perimeter): approximate wrist circumference
        wrist_width_px = calculate_distance(lm[0], lm[1])
        wrist_perimeter_px = wrist_width_px * 3.14  # Approximate circle perimeter

        # === Enregistrer l’image annotée ===
        annotated_filename = f"annotated_{os.path.basename(image_path)}"
        annotated_path = os.path.join("output", annotated_filename)
        cv2.imwrite(annotated_path, image)

        return {
            "longueur_main": round(pixels_to_mm(hand_length_px, coin_diameter_px, reference_mm), 2),
            "longueur_paume": round(pixels_to_mm(palm_length_px, coin_diameter_px, reference_mm), 2),
            "tour_main": round(pixels_to_mm(hand_perimeter_px, coin_diameter_px, reference_mm), 2),
            "tour_poignet": round(pixels_to_mm(wrist_perimeter_px, coin_diameter_px, reference_mm), 2),
            "annotated_image_filename": annotated_filename
        }
