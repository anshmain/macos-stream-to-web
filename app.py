from fastapi import FastAPI, Request, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from source.screen import screen

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="static")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("stream.html", {"request": request})

@app.websocket("/video")
async def video(websocket: WebSocket):
    await websocket.accept()
    for frame in screen():
        await websocket.send_bytes(frame)