import os
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import bikefit

app = FastAPI()
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)

status_speicher = {}
ergebnis_speicher = {}
abbruch_signale = {}
fortschritt_speicher = {}

@app.get("/health")
async def health_check():
    return {"status": "online"}

async def task_verarbeiten(dateiname: str, a_thigh: float, b_calf: float):
    in_pfad = f"uploads/{dateiname}"
    out_pfad = f"processed/out_{dateiname}"
    
    status_speicher[dateiname] = "Bearbeitung"
    
    try:
        abgeschlossen = await asyncio.to_thread(
            bikefit.analysiere_video,
            in_pfad,
            out_pfad,
            status_speicher,
            ergebnis_speicher,
            a_thigh,
            b_calf,
            abbruch_signale,
            fortschritt_speicher,
        )
    except Exception as exc:
        abgeschlossen = False
        status_speicher[dateiname] = "Fehler"
        ergebnis_speicher[dateiname] = f"Analyse fehlgeschlagen: {exc}"
    
    if status_speicher.get(dateiname) != "Fehler":
        status_speicher[dateiname] = "Fertig" if abgeschlossen else "Abgebrochen"
    
    if os.path.exists(in_pfad):
        os.remove(in_pfad)
        
    await asyncio.sleep(3600)
    
    if os.path.exists(out_pfad):
        os.remove(out_pfad)
    
    status_speicher.pop(dateiname, None)
    ergebnis_speicher.pop(dateiname, None)
    abbruch_signale.pop(dateiname, None)
    fortschritt_speicher.pop(dateiname, None)

@app.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks, 
    video: UploadFile = File(...),
    oberschenkel: float = Form(...),
    unterschenkel: float = Form(...)
):
    ziel = f"uploads/{video.filename}"
    abbruch_signale[video.filename] = False
    fortschritt_speicher[video.filename] = {
        "frames_analysiert": 0,
        "gesamt_frames": 0,
        "prozent": 0.0,
    }
    with open(ziel, "wb") as f:
        while chunk := await video.read(1024 * 1024):
            await asyncio.to_thread(f.write, chunk)
    
    background_tasks.add_task(task_verarbeiten, video.filename, oberschenkel, unterschenkel)
    return {"filename": video.filename, "message": "Start"}

@app.post("/cancel/{dateiname}")
async def cancel_video(dateiname: str):
    if dateiname not in abbruch_signale:
        return JSONResponse(status_code=404, content={"message": "Analyse nicht gefunden"})

    abbruch_signale[dateiname] = True
    status_speicher[dateiname] = "Abbruch angefordert"
    return {"filename": dateiname, "message": "Abbruch angefordert"}

@app.get("/status/{dateiname}")
async def get_status(dateiname: str):
    status = status_speicher.get(dateiname, "Unbekannt")
    befund = ergebnis_speicher.get(dateiname, "")
    fortschritt = fortschritt_speicher.get(dateiname, {
        "frames_analysiert": 0,
        "gesamt_frames": 0,
        "prozent": 0.0,
    })
    return JSONResponse(content={
        "status": status,
        "befund": befund,
        **fortschritt,
    })

@app.get("/video/{dateiname}")
async def get_video(dateiname: str):
    return FileResponse(f"processed/out_{dateiname}")

@app.get("/image/{dateiname}")
async def get_image(dateiname: str):
    bild_name = os.path.splitext(dateiname)[0]
    return FileResponse(f"processed/bild_{bild_name}.jpg")
