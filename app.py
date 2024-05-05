from fastapi import FastAPI, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from source.core.graphics.CG import CG
from source.modules.get_windows.macos.models import Windows
from source.screen import screen

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="static")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("stream.html", {"request": request})

@app.get("/windows", response_model_by_alias=False)
async def windows() -> Windows:
    return CG.windows()

@app.websocket("/video/{window_id}")
async def video(websocket: WebSocket, window_id: int):
    await websocket.accept()
    for frame in screen(window_id):
        await websocket.send_bytes(frame)