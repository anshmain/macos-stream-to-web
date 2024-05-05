# import zlib

# import cv2.data
# import numpy as np
# import Quartz.CoreGraphics as CG

# from source.core.compression.pako_deflate import pako_deflate


# def list_windows():
#     return CG.CGWindowListCopyWindowInfo(CG.kCGWindowListOptionOnScreenOnly, CG.kCGNullWindowID)

# def shot(options):
    

# def screen(window: int | None = None):
#     if window is None:
#         options = (
#             CG.CGRectInfinite,
#             CG.kCGWindowListOptionOnScreenOnly,
#             CG.kCGNullWindowID,
#             CG.kCGWindowImageNominalResolution
#         )
#     else:
#         options = (
#             CG.CGRectNull,
#             CG.kCGWindowListOptionIncludingWindow,
#             window,
#             CG.kCGWindowImageBoundsIgnoreFraming | CG.kCGWindowImageNominalResolution
#         )
#     while True:
#         buffer = shot(options)
#         yield pako_deflate(buffer.tobytes())
