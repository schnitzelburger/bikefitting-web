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

@app.get("/health")
async def health_check():
    return {"status": "online"}

async def task_verarbeiten(dateiname: str, a_thigh: float, b_calf: float):
    in_pfad = f"uploads/{dateiname}"
    out_pfad = f"processed/out_{dateiname}"
    
    status_speicher[dateiname] = "Bearbeitung"
    
    abgeschlossen = await asyncio.to_thread(
        bikefit.analysiere_video,
        in_pfad,
        out_pfad,
        status_speicher,
        ergebnis_speicher,
        a_thigh,
        b_calf,
        abbruch_signale,
    )
    
    status_speicher[dateiname] = "Fertig" if abgeschlossen else "Abgebrochen"
    
    if os.path.exists(in_pfad):
        os.remove(in_pfad)
        
    await asyncio.sleep(3600)
    
    if os.path.exists(out_pfad):
        os.remove(out_pfad)
    
    status_speicher.pop(dateiname, None)
    ergebnis_speicher.pop(dateiname, None)
    abbruch_signale.pop(dateiname, None)

@app.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks, 
    video: UploadFile = File(...),
    oberschenkel: float = Form(...),
    unterschenkel: float = Form(...)
):
    ziel = f"uploads/{video.filename}"
    abbruch_signale[video.filename] = False
    with open(ziel, "wb") as f:
        while chunk := await video.read(1024 * 1024):
            f.write(chunk)
    
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
    return JSONResponse(content={"status": status, "befund": befund})

@app.get("/video/{dateiname}")
async def get_video(dateiname: str):
    return FileResponse(f"processed/out_{dateiname}")

@app.get("/image/{dateiname}")
async def get_image(dateiname: str):
    return FileResponse(f"processed/bild_{dateiname}.jpg")
