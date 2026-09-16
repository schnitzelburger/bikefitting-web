from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import os
import httpx

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
    async with httpx.AsyncClient(timeout=60.0) as client:
        files = {"video": (video.filename, await video.read(), video.content_type)}
        data = {"oberschenkel": oberschenkel, "unterschenkel": unterschenkel}
        await client.post(f"{WORKER_URL}/analyze", files=files, data=data, timeout=60.0)
    
    return HTMLResponse(content=f'''
        <div id="analysestatus" hx-get="/status/{video.filename}" hx-trigger="every 2s" hx-swap="outerHTML">
            <p>Analyse gestartet. Bitte warten...</p>
        </div>
    ''')

@app.get("/status/{dateiname}", response_class=HTMLResponse)
async def get_status(dateiname: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{WORKER_URL}/status/{dateiname}")
        data = response.json()
        
    status = data.get("status")
    befund = data.get("befund")
    
    if status == "Fertig":
        return HTMLResponse(content=f'''
            <div id="analysestatus">
                <h3>Ergebnis</h3>
                <p style="white-space: pre-wrap; font-weight: bold; background: #eee; padding: 15px; border-radius: 5px;">{befund}</p>
                <img src="/image/{dateiname}.jpg" width="100%">
                <video width="100%" controls>
                    <source src="/video/{dateiname}" type="video/mp4">
                </video>
            </div>
        ''')
        
    return HTMLResponse(content=f'''
        <div id="analysestatus" hx-get="/status/{dateiname}" hx-trigger="every 2s" hx-swap="outerHTML">
            <p>Aktueller Status: {status}</p>
        </div>
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
