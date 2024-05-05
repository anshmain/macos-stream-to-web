from functools import cache
from typing import Any

import cv2
import numpy as np

from source.core.graphics.AbstractScreen import AbstractScreen
from source.core.graphics.CG import CG, ImageOptions
from source.modules.graphics.macos.models import Resolution


class Screen(AbstractScreen):
    
    @cache
    def __build_options(self, window: int | None) -> ImageOptions:
        return ImageOptions(
            rect=CG.rect_null() if window else CG.rect_infinite(),
            list_option=CG.list_option_including_window() if window else CG.list_option_on_screen_only(),
            window_id=window or CG.null_window_id(),
            image_option=CG.bounds_ignore_framing() or CG.nominal_resolution() if window else CG.nominal_resolution()
        )

    def get(self, resolution: Resolution, window: int | None) -> np.ndarray[Any, np.dtype[np.uint8]]:
        image = CG.create_image(self.__build_options(window))

        np_raw_data = np.frombuffer(image.data, dtype=np.uint8)
        np_data = np.lib.stride_tricks.as_strided(np_raw_data,
                                                    shape=(image.metadata.height, image.metadata.width, 3),
                                                    strides=(image.metadata.bpr, 4, 1),
                                                    writeable=False)

        resized = cv2.resize(np_data, resolution)
        _, buffer = cv2.imencode('.jpg', resized)
        return buffer