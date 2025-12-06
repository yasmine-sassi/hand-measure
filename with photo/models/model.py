# Télécharger automatiquement si le fichier n'existe pas
import os
import requests

if not os.path.exists("best.pt"):
    url = "https://drive.google.com/file/d/1swudXR1hu7SL7-KzkFGrIYwRXSPcCsto/view?usp=sharing"
    r = requests.get(url)
    with open("best.pt", "wb") as f:
        f.write(r.content)
