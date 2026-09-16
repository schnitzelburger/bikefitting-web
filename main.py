from fastapi import FastAPI, UploadFile, File, Request, BackgroundTasks
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

def starte_analyse(dateiname: str):
    pfad = f"static/uploads/{dateiname}"
    status_speicher[dateiname] = "Vorbereitung"
    bikefit.analysiere_video(pfad, dateiname, status_speicher)
    status_speicher[dateiname] = "Fertig"

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/upload", response_class=HTMLResponse)
async def upload_video(request: Request, background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    dateipfad = f"static/uploads/{video.filename}"
    with open(dateipfad, "wb") as f:
        f.write(await video.read())
    
    background_tasks.add_task(starte_analyse, video.filename)
    
    return HTMLResponse(content=f'''
        <div hx-get="/status/{video.filename}" hx-trigger="every 2s">
            <p>Analyse gestartet. Bitte warten...</p>
        </div>
    ''')

@app.get("/status/{dateiname}", response_class=HTMLResponse)
async def get_status(dateiname: str):
    status = status_speicher.get(dateiname, "Verarbeite")
    if status == "Fertig":
        return HTMLResponse(content=f'''
            <h3>Ergebnis</h3>
            <video width="100%" controls>
                <source src="/static/uploads/out_{dateiname}" type="video/mp4">
            </video>
        ''')
    return HTMLResponse(content=f'''
        <div hx-get="/status/{dateiname}" hx-trigger="every 2s">
            <p>Aktueller Status: {status}</p>
        </div>
    ''')
