from typing import Any, Generic, NamedTuple, TypeVar

import Quartz.CoreGraphics as QCG  # type: ignore
from pydantic import BaseModel

from source.modules.get_windows.macos.models import Window, Windows


class ImageMetadata(BaseModel):
    bpr: int
    width: int
    height: int


class ImageCG:
    def __init__(self, image: Any): # type: ignore
        self.__image = image
    
    @property
    def data(self) -> bytes:
        return QCG.CGDataProviderCopyData(QCG.CGImageGetDataProvider(self.__image)) # type: ignore
    
    @property
    def metadata(self) -> ImageMetadata:
        return ImageMetadata(
            bpr=QCG.CGImageGetBytesPerRow(self.__image), # type: ignore
            width=QCG.CGImageGetWidth(self.__image), # type: ignore
            height=QCG.CGImageGetHeight(self.__image) # type: ignore
        )

# Rect
RectInfinite = TypeVar('RectInfinite')
RectNull = TypeVar('RectNull')

# Window List Option
OnScreenOnly = TypeVar('OnScreenOnly')
IncludingWindow = TypeVar('IncludingWindow')

# Window ID
NullWindow = TypeVar('NullWindow')

# Image Option
NominalResolution = TypeVar('NominalResolution')
BoundsIgnoreFraming = TypeVar('BoundsIgnoreFraming')


class ImageOptions(
    NamedTuple,
    Generic[RectInfinite, RectNull, OnScreenOnly, IncludingWindow, NullWindow, NominalResolution, BoundsIgnoreFraming]
):
    rect: RectInfinite | RectNull
    list_option: OnScreenOnly | IncludingWindow
    window_id: int | NullWindow
    image_option: NominalResolution | BoundsIgnoreFraming


class CG(Generic[RectInfinite, RectNull, OnScreenOnly, IncludingWindow, NullWindow, NominalResolution, BoundsIgnoreFraming]):
    @classmethod
    def create_image(cls, options: ImageOptions) -> ImageCG:
        return ImageCG(image=QCG.CGWindowListCreateImage(*options)) # type: ignore
    
    @classmethod
    def rect_infinite(cls) -> RectInfinite:
        return QCG.CGRectInfinite # type: ignore
    
    @classmethod
    def rect_null(cls) -> RectNull:
        return QCG.CGRectNull # type: ignore
    
    @classmethod
    def list_option_on_screen_only(cls) -> OnScreenOnly:
        return QCG.kCGWindowListOptionOnScreenOnly # type: ignore
    
    @classmethod
    def list_option_including_window(cls) -> IncludingWindow:
        return QCG.kCGWindowListOptionIncludingWindow # type: ignore
    
    @classmethod
    def null_window_id(cls) -> NullWindow:
        return QCG.kCGNullWindowID # type: ignore
    
    @classmethod
    def nominal_resolution(cls) -> NominalResolution:
        return QCG.kCGWindowImageNominalResolution # type: ignore
    
    @classmethod
    def bounds_ignore_framing(cls) -> BoundsIgnoreFraming:
        return QCG.kCGWindowImageNominalResolution # type: ignore
    
    @classmethod
    def __dump_window(cls, w: Any) -> dict | None:
        dumped = dict(w)
        if not (dumped['kCGWindowName'] and dumped['kCGWindowNumber']):
            return None
        dumped['kCGWindowBounds'] = dict(dumped['kCGWindowBounds'])
        return dumped

    @classmethod
    def windows(cls) -> Windows:
        return Windows(
            windows=[
                Window.model_validate(w)
                for w in map(cls.__dump_window, QCG.CGWindowListCopyWindowInfo(QCG.kCGWindowListOptionOnScreenOnly, QCG.kCGNullWindowID)) if w # type: ignore
            ]
        )
