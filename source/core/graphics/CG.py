from typing import Generic, NamedTuple, TypeVar

import Quartz.CoreGraphics as CG  # type: ignore
from pydantic import BaseModel


class ImageMetadata(BaseModel):
    bpr: int
    width: int
    height: int


class ImageCG:
    def __init__(self, image: CG.CGImage): # type: ignore
        self.__image = image
    
    @property
    def data(self) -> bytes:
        return CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(self.__image)) # type: ignore
    
    @property
    def metadata(self) -> ImageMetadata:
        return ImageMetadata(
            bpr=CG.CGImageGetBytesPerRow(self.__image), # type: ignore
            width=CG.CGImageGetWidth(self.__image), # type: ignore
            height=CG.CGImageGetHeight(self.__image) # type: ignore
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
        return ImageCG(image=CG.CGWindowListCreateImage(*options)) # type: ignore
    
    @classmethod
    def rect_infinite(cls) -> RectInfinite:
        return CG.CGRectInfinite # type: ignore
    
    @classmethod
    def rect_null(cls) -> RectNull:
        return CG.CGRectNull # type: ignore
    
    @classmethod
    def list_option_on_screen_only(cls) -> OnScreenOnly:
        return CG.kCGWindowListOptionOnScreenOnly # type: ignore
    
    @classmethod
    def list_option_including_window(cls) -> IncludingWindow:
        return CG.kCGWindowListOptionIncludingWindow # type: ignore
    
    @classmethod
    def null_window_id(cls) -> NullWindow:
        return CG.kCGNullWindowID # type: ignore
    
    @classmethod
    def nominal_resolution(cls) -> NominalResolution:
        return CG.kCGWindowImageNominalResolution # type: ignore
    
    @classmethod
    def bounds_ignore_framing(cls) -> BoundsIgnoreFraming:
        return CG.kCGWindowImageNominalResolution # type: ignore
