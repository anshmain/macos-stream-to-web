from time import time
import zlib

import cv2.data
import numpy as np
import Quartz.CoreGraphics as CG


def pako_deflate_raw(data):
    compressed_data = zlib.compress(data, level=9)
    return compressed_data


def screen():
    windows = CG.CGWindowListCopyWindowInfo(CG.kCGWindowListOptionOnScreenOnly, CG.kCGNullWindowID)
    win_id = CG.kCGNullWindowID
    for window in windows:
        if window.get('kCGWindowOwnerName') == '':
            win_id = window.get('kCGWindowNumber')
    print(win_id)
    # screen_only = CG.kCGWindowListOptionOnScreenOnly
    # null_window = CG.kCGNullWindowID
    # nominal_resolution = CG.kCGWindowImageNominalResolution
    # kCGWindowListOptionIncludingWindow
    # kCGWindowImageBoundsIgnoreFraming
    while True:
        t = time()
        frame = CG.CGWindowListCreateImage(
            CG.CGRectNull,
            CG.kCGWindowListOptionIncludingWindow,
            1358,
            CG.kCGWindowImageBoundsIgnoreFraming | CG.kCGWindowImageNominalResolution,
        )

        bpr = CG.CGImageGetBytesPerRow(frame)
        width = CG.CGImageGetWidth(frame)
        height = CG.CGImageGetHeight(frame)

        np_raw_data = np.frombuffer(CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(frame)), dtype=np.uint8)
        np_data = np.lib.stride_tricks.as_strided(np_raw_data,
                                                  shape=(height, width, 3),
                                                  strides=(bpr, 4, 1),
                                                  writeable=False)

        resized = cv2.resize(np_data, (1920, 1080))
        _, buffer = cv2.imencode('.jpg', resized)
        yield pako_deflate_raw(buffer.tobytes())
        print(time() - t)
