from pydantic import BaseModel, Field


class Bounds(BaseModel):
    h: float = Field(alias='Height')
    w: float = Field(alias='Width')
    x: float = Field(alias='X')
    y: float = Field(alias='Y')


class Window(BaseModel):
    alpha: float = Field(alias='kCGWindowAlpha')
    bounds: Bounds = Field(alias='kCGWindowBounds')
    is_on_screen: bool = Field(alias='kCGWindowIsOnscreen')
    layer: float = Field(alias='kCGWindowLayer')
    mem_usage: float = Field(alias='kCGWindowMemoryUsage')
    name: str = Field(alias='kCGWindowName')
    number: int = Field(alias='kCGWindowNumber')
    owner_name: str = Field(alias='kCGWindowOwnerName')
    owner_pid: int = Field(alias='kCGWindowOwnerPID')
    sharing_state: int = Field(alias='kCGWindowSharingState')
    store_type: int = Field(alias='kCGWindowStoreType')


class Windows(BaseModel):
    windows: list[Window]
