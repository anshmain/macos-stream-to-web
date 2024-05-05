from source.core.compression.pako_deflate import pako_deflate
from source.modules.graphics.macos.models import Resolution


def screen(window: int | None = None, os: str = 'macos', resolution: Resolution = Resolution(width=1280, height=720)):
    if os == 'macos':
        from source.modules.graphics.macos import Screen
    elif os == 'windows':
        from source.modules.graphics.macos import Screen
        pass
    else:
        from source.modules.graphics.macos import Screen
    
    s = Screen()
    while True:
        buffer = s.get(resolution, window)
        yield pako_deflate(buffer.tobytes())
