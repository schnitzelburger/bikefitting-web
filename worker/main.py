from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import os
import bikefit

app = FastAPI()
os.makedirs("uploads", exist_ok=True)

status_speicher = {}
ergebnis_speicher = {}

def starte_analyse(dateiname: str, a_thigh: float, b_calf: float):
    pfad = f"uploads/{dateiname}"
    status_speicher[dateiname] = "Vorbereitung"
    bikefit.analysiere_video(pfad, dateiname, status_speicher, ergebnis_speicher, a_thigh, b_calf)
    status_speicher[dateiname] = "Fertig"

@app.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks, 
    video: UploadFile = File(...),
    oberschenkel: float = Form(...),
    unterschenkel: float = Form(...)
):
    dateipfad = f"uploads/{video.filename}"
    with open(dateipfad, "wb") as f:
        f.write(await video.read())
    
    background_tasks.add_task(starte_analyse, video.filename, oberschenkel, unterschenkel)
    return {"filename": video.filename, "message": "Analyse gestartet"}

@app.get("/status/{dateiname}")
async def get_status(dateiname: str):
    status = status_speicher.get(dateiname, "Unbekannt")
    befund = ergebnis_speicher.get(dateiname, "")
    return JSONResponse(content={"status": status, "befund": befund})

@app.get("/video/{dateiname}")
async def get_video(dateiname: str):
    return FileResponse(f"uploads/out_{dateiname}")
