import numpy as np
import numpy.typing as npt

from .run import Apu, Pulse, Noise, Triangle

def apu_mixer(
    apu: Apu,
    pulse1: npt.NDArray[np.uint8],
    pulse2: npt.NDArray[np.uint8],
    triangle: npt.NDArray[np.uint8],
    noise: npt.NDArray[np.uint8],
    dmc: npt.NDArray[np.uint8],
    output: npt.NDArray[np.int16],
) -> None: ...
def generate_pulse(
    pulse: Pulse,
    pulse_out: npt.NDArray[np.uint8],
) -> None: ...
def generate_triangle(
    triangle: Triangle,
    triangle_out: npt.NDArray[np.uint8],
) -> None: ...
def generate_noise(
    noise: Noise,
    noise_out: npt.NDArray[np.uint8],
) -> None: ...
