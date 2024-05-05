from abc import ABC, abstractmethod


class AbstractScreen(ABC):
    
    @abstractmethod
    def get(self, resolution: tuple[int, int], window: int | None) -> bytes:
        pass