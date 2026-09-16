from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import bikefit

app = FastAPI()

os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

status_speicher = {}
ergebnis_speicher = {}

def starte_analyse(dateiname: str, a_thigh: float, b_calf: float):
    pfad = f"static/uploads/{dateiname}"
    status_speicher[dateiname] = "Vorbereitung"
    bikefit.analysiere_video(pfad, dateiname, status_speicher, ergebnis_speicher, a_thigh, b_calf)
    status_speicher[dateiname] = "Fertig"

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/upload", response_class=HTMLResponse)
async def upload_video(
    request: Request, 
    background_tasks: BackgroundTasks, 
    video: UploadFile = File(...),
    oberschenkel: float = Form(...),
    unterschenkel: float = Form(...)
):
    dateipfad = f"static/uploads/{video.filename}"
    with open(dateipfad, "wb") as f:
        f.write(await video.read())
    
    background_tasks.add_task(starte_analyse, video.filename, oberschenkel, unterschenkel)
    
    return HTMLResponse(content=f'''
        <div id="analysestatus" hx-get="/status/{video.filename}" hx-trigger="every 2s" hx-swap="outerHTML">
            <p>Analyse gestartet. Bitte warten...</p>
        </div>
    ''')

@app.get("/status/{dateiname}", response_class=HTMLResponse)
async def get_status(dateiname: str):
    status = status_speicher.get(dateiname, "Verarbeite")
    
    if status == "Fertig":
        befund = ergebnis_speicher.get(dateiname, "Analyse abgeschlossen.")
        return HTMLResponse(content=f'''
            <div id="analysestatus">
                <h3>Ergebnis</h3>
                <p style="white-space: pre-wrap; font-weight: bold; background: #eee; padding: 15px; border-radius: 5px;">{befund}</p>
                <video width="100%" controls>
                    <source src="/static/uploads/out_{dateiname}" type="video/mp4">
                </video>
            </div>
        ''')
        
    return HTMLResponse(content=f'''
        <div id="analysestatus" hx-get="/status/{dateiname}" hx-trigger="every 2s" hx-swap="outerHTML">
            <p>Aktueller Status: {status}</p>
        </div>
    ''')