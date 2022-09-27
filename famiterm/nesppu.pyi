import numpy as np
import numpy.typing as npt

def get_color(index: int) -> int: ...
def blit(
    source: npt.NDArray[np.uint32],
    destination: npt.NDArray[np.uint32],
    coordinate: tuple[int, int],
) -> None: ...
def render_tile(
    rom: bytes,
    address: int,
    colors: bytes,
    destination: npt.NDArray[np.uint32],
) -> None: ...
