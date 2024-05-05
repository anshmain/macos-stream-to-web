import zlib


def pako_deflate(data: bytes, level: int = 9) -> bytes:
    return zlib.compress(data, level=level)
