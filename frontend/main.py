from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import os
import httpx
import tempfile
import asyncio
import html
import re

app = FastAPI()
templates = Jinja2Templates(directory="templates")

WORKER_URL = os.getenv("WORKER_URL", "http://192.168.178.50:8000")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/worker-status", response_class=HTMLResponse)
async def get_worker_status():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{WORKER_URL}/health", timeout=3.0)
            response.raise_for_status()
        return HTMLResponse(content='<div class="worker-status online"><span class="status-dot"></span><span>Worker online</span></div>')
    except (httpx.HTTPError, OSError):
        return HTMLResponse(content='<div class="worker-status offline"><span class="status-dot"></span><span>Worker offline</span></div>')

@app.post("/upload", response_class=HTMLResponse)
async def upload_video(
    video: UploadFile = File(...),
    oberschenkel: float = Form(...),
    unterschenkel: float = Form(...)
):
    with tempfile.NamedTemporaryFile() as upload_file:
        while chunk := await video.read(1024 * 1024):
            await asyncio.to_thread(upload_file.write, chunk)
        upload_file.flush()
        upload_file.seek(0)
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"video": (video.filename, upload_file, video.content_type)}
            data = {"oberschenkel": oberschenkel, "unterschenkel": unterschenkel}
            response = await client.post(f"{WORKER_URL}/analyze", files=files, data=data, timeout=60.0)
            response.raise_for_status()
    
    return HTMLResponse(content=f'''
        <div id="analyse-container">
            <div id="analysestatus" hx-get="/status/{video.filename}" hx-trigger="every 2s" hx-swap="outerHTML">
                <p class="analyse-status-text">Analyse wird vorbereitet...</p>
                <progress class="analyse-progress" value="0" max="100"></progress>
            </div>
        </div>
        <button id="abbrechen-button" type="button"
            hx-post="/cancel/{video.filename}" hx-target="#analyse-container" hx-swap="outerHTML"
                hx-swap-oob="true">Analyse abbrechen</button>
    ''')

@app.get("/status/{dateiname}", response_class=HTMLResponse)
async def get_status(dateiname: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{WORKER_URL}/status/{dateiname}")
        data = response.json()
        
    status = data.get("status")
    befund = data.get("befund")
    percent = data.get("prozent", 0.0)
    frames = data.get("frames_analysiert", 0)
    total_frames = data.get("gesamt_frames", 0)

    if status == "Fehler":
        return HTMLResponse(content=f'''
            <div id="analysestatus">
                <h3>Analyse fehlgeschlagen</h3>
                <p>{befund}</p>
            </div>
            <button id="abbrechen-button" type="button" disabled hx-swap-oob="true">Analyse nicht verfügbar</button>
        ''')
    
    if status == "Fertig":
        rumpf = re.search(r"Rumpf liegt bei ([0-9.]+) Grad", befund or "")
        ellbogen = re.search(r"Ellbogen bei ([0-9.]+) Grad", befund or "")
        schulter = re.search(r"Schulterwinkel: ([0-9.]+) Grad", befund or "")
        huefte = re.search(r"Hueftwinkel: ([0-9.]+) Grad", befund or "")
        messwerte = {
            "Haltung des Rumpfs": rumpf.group(1) if rumpf else "-",
            "Ellbogen": ellbogen.group(1) if ellbogen else "-",
            "Schulterwinkel": schulter.group(1) if schulter else "-",
            "Hüftwinkel": huefte.group(1) if huefte else "-",
        }
        messwerte_html = "".join(
            f'<div><strong>{html.escape(name)}</strong>: {html.escape(value)} Grad</div>'
            for name, value in messwerte.items()
        )
        empfehlung = re.split(r"\n\nHaltung", befund or "", maxsplit=1)[0]
        return HTMLResponse(content=f'''
            <div id="analysestatus">
                <h3>Ergebnis</h3>
                <p style="white-space: pre-wrap; font-weight: bold; background: #eee; padding: 15px; border-radius: 5px;">{html.escape(empfehlung)}</p>
                <h3>Messwerte</h3>
                <div>{messwerte_html}</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; align-items: start;">
                    <div>
                        <h4>Totpunktbild</h4>
                        <img src="/image/{dateiname}" style="display: block; width: 100%; height: auto;" alt="Totpunktbild der Bikefit-Analyse">
                    </div>
                    <div>
                        <h4>Analysiertes Video</h4>
                        <video style="display: block; width: 100%; height: auto;" controls>
                            <source src="/video/{dateiname}" type="video/mp4">
                        </video>
                    </div>
                </div>
            </div>
            <button id="abbrechen-button" type="button" disabled hx-swap-oob="true">Analyse abgeschlossen</button>
        ''')

    if status in {"Abgebrochen", "Abbruch angefordert"}:
        return HTMLResponse(content=f'''
            <div id="analysestatus">
                <p>Analyse wurde abgebrochen.</p>
            </div>
            <button id="abbrechen-button" type="button" disabled hx-swap-oob="true">Analyse abgebrochen</button>
        ''')
        
    return HTMLResponse(content=f'''
        <div id="analysestatus" hx-get="/status/{dateiname}" hx-trigger="every 2s" hx-swap="outerHTML">
            <p class="analyse-status-text">Verarbeite Frame {frames} von {total_frames} ({percent:.2f} %)</p>
            <progress class="analyse-progress" value="{percent}" max="100"></progress>
        </div>
    ''')

@app.post("/cancel/{dateiname}", response_class=HTMLResponse)
async def cancel_video(dateiname: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{WORKER_URL}/cancel/{dateiname}", timeout=10.0)
        response.raise_for_status()

    return HTMLResponse(content=f'''
        <div id="analyse-container">
            <div id="analysestatus">
                <p>Abbruch angefordert. Die laufende Verarbeitung wird beendet.</p>
            </div>
        </div>
        <button id="abbrechen-button" type="button" disabled hx-swap-oob="true">Abbruch angefordert</button>
    ''')

@app.get("/video/{dateiname}")
async def get_video(dateiname: str):
    async def stream_video():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", f"{WORKER_URL}/video/{dateiname}") as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
                    
    return StreamingResponse(stream_video(), media_type="video/mp4")

@app.get("/image/{dateiname}")
async def get_image(dateiname: str):
    async def stream_image():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", f"{WORKER_URL}/image/{dateiname}") as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
    return StreamingResponse(stream_image(), media_type="image/jpeg")
