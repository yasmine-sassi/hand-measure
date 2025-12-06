uvicorn main:app --host 0.0.0.0 --port 8000

✅ Complete Integration
Backend (Python):

Connected to MySQL database (dashboard)
Retrieves campaign_id from React app
Queries campaign_referentiel and referentiel tables
Measures hand dimensions: longueur_main, longueur_paume, tour_main, tour_poignet
Matches measurements against referentiel criteria
Returns recommended glove size based on database values
Frontend (React):

Sends campaign_id from URL to Python backend
Captures hand image and sends to /detect/ endpoint
Receives measurements and recommended sizes
Displays size recommendations with "Recommandé" and "Ample" options
Database-Driven:

Size recommendations now come entirely from your referentiel data
No hardcoded size logic
Fully configurable through database

Extracts all 4 criteria from the referentiel value JSON:

longueur_main (hand/finger length)
longueur_paume (palm width)
tour_main (hand perimeter)
tour_poignet (wrist perimeter)
Compares measured values against each available size (5, 6, 7, 8, 9, 10, 11, 12, etc.)

Calculates a weighted score:

tour_main is weighted 2x (most important for glove fit)
Other criteria weighted 1x each
Recommends the size with the lowest average difference across all criteria
