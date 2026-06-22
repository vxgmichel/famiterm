from __future__ import annotations
from argparse import ArgumentParser, Namespace

from pathlib import Path
import pickle
from copy import deepcopy
from enum import IntEnum
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Callable
import zlib


import numpy as np
import numpy.typing as npt
from gambaterm.console import Console
from gambaterm.main import main as gambaterm_main

from . import nescpu
from . import nesppu
from . import nesapu


class InfiniteLoop(Exception):
    pass


class InstructionBudgetExhausted(Exception):
    pass


class UnsupportedCartridgeError(RuntimeError):
    pass


@dataclass
class Cartridge:
    mapper: int
    mirroring: str
    cartridge_has_prg_ram: bool
    has_trainer: bool
    ignore_mirroring_control: bool
    trainer: bytes | None
    prg_rom: bytes
    chr_rom: bytes
    prg_ram: bytearray = field(default_factory=lambda: bytearray(8 * 1024))
    chr_ram: bytearray | None = None
    mapped_prg_rom: bytes = b""
    mmc1_shift_register: int = 0x10
    mmc1_control: int = 0x0C
    mmc1_chr_bank0: int = 0
    mmc1_chr_bank1: int = 0
    mmc1_prg_bank: int = 0

    def __post_init__(self) -> None:
        if self.chr_ram is None and not self.chr_rom:
            self.chr_ram = bytearray(8 * 1024)
        self.update_mirroring()
        self.update_mapped_prg_rom()

    def update_mirroring(self) -> None:
        if self.mapper != 1:
            return
        mode = self.mmc1_control & 0x03
        if mode == 0:
            self.mirroring = "1L"
        elif mode == 1:
            self.mirroring = "1H"
        elif mode == 2:
            self.mirroring = "V"
        else:
            self.mirroring = "H"

    @property
    def prg_bank_count(self) -> int:
        return len(self.prg_rom) // (16 * 1024)

    @property
    def chr_bank_count(self) -> int:
        if self.chr_rom:
            return len(self.chr_rom) // (4 * 1024)
        assert self.chr_ram is not None
        return len(self.chr_ram) // (4 * 1024)

    def _prg_offset(self, bank: int, offset: int) -> int:
        return ((bank % self.prg_bank_count) * 16 * 1024) + offset

    def prg_bank(self, bank: int) -> bytes:
        start = self._prg_offset(bank, 0)
        return self.prg_rom[start : start + 16 * 1024]

    def update_mapped_prg_rom(self) -> None:
        if self.mapper == 0:
            self.mapped_prg_rom = self.prg_rom
            return
        if self.mapper == 1:
            prg_mode = (self.mmc1_control >> 2) & 0x03
            if prg_mode in (0, 1):
                bank = self.mmc1_prg_bank & ~0x01
                self.mapped_prg_rom = self.prg_bank(bank) + self.prg_bank(bank + 1)
            elif prg_mode == 2:
                self.mapped_prg_rom = self.prg_bank(0) + self.prg_bank(self.mmc1_prg_bank)
            else:
                self.mapped_prg_rom = self.prg_bank(self.mmc1_prg_bank) + self.prg_bank(
                    self.prg_bank_count - 1
                )
            return
        raise UnsupportedCartridgeError(f"Unsupported NES mapper {self.mapper}")

    def _chr_storage(self) -> bytes | bytearray:
        return self.chr_rom if self.chr_rom else self.chr_ram

    @property
    def chr_cache_key(self) -> tuple[int, int, int, int]:
        if self.mapper == 1:
            return (
                self.mapper,
                self.mmc1_control & 0x10,
                self.mmc1_chr_bank0,
                self.mmc1_chr_bank1,
            )
        return (self.mapper, 0, 0, 0)

    def _chr_offset(self, bank: int, offset: int) -> int:
        return ((bank % self.chr_bank_count) * 4 * 1024) + offset

    def cpu_read(self, addr: int) -> int:
        if 0x6000 <= addr < 0x8000:
            return self.prg_ram[addr - 0x6000]
        if 0x8000 <= addr < 0x10000:
            if self.mapper == 0:
                return self.prg_rom[addr - 0x8000]
            if self.mapper == 1:
                prg_mode = (self.mmc1_control >> 2) & 0x03
                offset = addr & 0x3FFF
                if prg_mode in (0, 1):
                    bank = self.mmc1_prg_bank & ~0x01
                    if addr >= 0xC000:
                        bank += 1
                elif prg_mode == 2:
                    bank = 0 if addr < 0xC000 else self.mmc1_prg_bank
                else:
                    bank = self.mmc1_prg_bank if addr < 0xC000 else self.prg_bank_count - 1
                return self.prg_rom[self._prg_offset(bank, offset)]
        raise ValueError(f"Invalid cartridge read: 0x{addr:04x}")

    def cpu_write(self, addr: int, value: int) -> None:
        if 0x6000 <= addr < 0x8000:
            self.prg_ram[addr - 0x6000] = value
            return
        if 0x8000 <= addr < 0x10000:
            if self.mapper == 0:
                return
            if self.mapper == 1:
                self.write_mmc1_register(addr, value)
                return
        raise ValueError(f"Invalid cartridge write: 0x{addr:04x}")

    def write_mmc1_register(self, addr: int, value: int) -> None:
        if value & 0x80:
            self.mmc1_shift_register = 0x10
            self.mmc1_control |= 0x0C
            self.update_mirroring()
            self.update_mapped_prg_rom()
            return
        complete = self.mmc1_shift_register & 0x01
        self.mmc1_shift_register >>= 1
        self.mmc1_shift_register |= (value & 0x01) << 4
        if not complete:
            return
        data = self.mmc1_shift_register & 0x1F
        self.mmc1_shift_register = 0x10
        if 0x8000 <= addr < 0xA000:
            self.mmc1_control = data
            self.update_mirroring()
            self.update_mapped_prg_rom()
        elif 0xA000 <= addr < 0xC000:
            self.mmc1_chr_bank0 = data
        elif 0xC000 <= addr < 0xE000:
            self.mmc1_chr_bank1 = data
        else:
            self.mmc1_prg_bank = data & 0x0F
            self.update_mapped_prg_rom()

    def ppu_read(self, addr: int) -> int:
        addr &= 0x1FFF
        if self.mapper == 1:
            chr_mode = (self.mmc1_control >> 4) & 0x01
            if chr_mode == 0:
                bank = self.mmc1_chr_bank0 & ~0x01
                offset = addr & 0x0FFF
                if addr >= 0x1000:
                    bank += 1
            else:
                bank = self.mmc1_chr_bank0 if addr < 0x1000 else self.mmc1_chr_bank1
                offset = addr & 0x0FFF
            storage = self._chr_storage()
            return storage[self._chr_offset(bank, offset)]
        storage = self._chr_storage()
        return storage[addr % len(storage)]

    def ppu_write(self, addr: int, value: int) -> None:
        if self.chr_ram is None:
            return
        addr &= 0x1FFF
        if self.mapper == 1:
            chr_mode = (self.mmc1_control >> 4) & 0x01
            if chr_mode == 0:
                bank = self.mmc1_chr_bank0 & ~0x01
                offset = addr & 0x0FFF
                if addr >= 0x1000:
                    bank += 1
            else:
                bank = self.mmc1_chr_bank0 if addr < 0x1000 else self.mmc1_chr_bank1
                offset = addr & 0x0FFF
            self.chr_ram[self._chr_offset(bank, offset)] = value
        else:
            self.chr_ram[addr % len(self.chr_ram)] = value


class ApuRegister(IntEnum):
    PULSE1_CONFIG = 0x00
    PULSE1_SWEEP = 0x01
    PULSE1_TIMER = 0x02
    PULSE1_LENGTH_COUNTER = 0x03
    PULSE2_CONFIG = 0x04
    PULSE2_SWEEP = 0x05
    PULSE2_TIMER = 0x06
    PULSE2_LENGTH_COUNTER = 0x07
    TRIANGLE_CONFIG = 0x08
    TRIANGLE_UNUSED = 0x09
    TRIANGLE_TIMER = 0x0A
    TRIANGLE_LENGTH_COUNTER = 0x0B
    NOISE_CONFIG = 0x0C
    NOISE_UNUSED = 0x0D
    NOISE_PERIOD = 0x0E
    NOISE_LENGTH_COUNTER = 0x0F
    DMC_CONFIG = 0x10
    DMC_LOAD_COUNTER = 0x11
    DMC_SAMPLE_ADDRESS = 0x12
    DMC_SAMPLE_LENGTH = 0x13
    STATUS = 0x15
    FRAME_COUNTER = 0x17


APU_LENGTH_TABLE = [
    10,
    254,
    20,
    2,
    40,
    4,
    80,
    6,
    160,
    8,
    60,
    10,
    14,
    12,
    26,
    14,
    12,
    16,
    24,
    18,
    48,
    20,
    96,
    22,
    192,
    24,
    72,
    26,
    16,
    28,
    32,
    30,
]
assert len(APU_LENGTH_TABLE) == 32


@dataclass(eq=False)
class Pulse:
    id: int
    enabled: bool = False
    duty: int = 0
    length_counter_halt: bool = False
    constant_volume: bool = False
    volume: int = 0
    sweep_enabled: bool = False
    sweep_period: int = 0
    sweep_negate: bool = False
    sweep_shift_count: int = 0
    load_timer: int = 0
    load_length_counter: int = 0

    # Internal state
    start_flag: int = 0
    current_tick: int = 0
    current_timer: int = 0
    sweep_divider: int = 0
    length_counter: int = 0
    divider_period: int = 0
    current_sequencer: int = 0
    sweep_reload_flag: int = 0
    decay_level_counter: int = 0
    current_timer_period: int = 0

    def set_enabled(self, value: bool) -> None:
        self.enabled = value
        if not value:
            self.length_counter = 0

    def write_register(self, register: ApuRegister, value: int) -> None:
        if (register & ~0x04) == register.PULSE1_CONFIG:
            self.duty = value >> 6
            self.length_counter_halt = bool(value & 0x20)
            self.constant_volume = bool(value & 0x10)
            self.volume = value & 0xF
            return
        if (register & ~0x04) == register.PULSE1_SWEEP:
            self.sweep_enabled = bool(value & 0x80)
            self.sweep_period = ((value >> 4) & 0x07) + 1
            self.sweep_negate = bool(value & 0x08)
            self.sweep_shift_count = value & 0x07
            # Side effects
            self.sweep_reload_flag = 1
            return
        if (register & ~0x04) == register.PULSE1_TIMER:
            self.load_timer &= ~0xFF
            self.load_timer |= value
            return
        if (register & ~0x04) == register.PULSE1_LENGTH_COUNTER:
            self.load_timer &= ~0x700
            self.load_timer |= (value & 0x7) << 8
            self.load_length_counter = value >> 3
            # Side effects
            if self.enabled:
                self.length_counter = APU_LENGTH_TABLE[self.load_length_counter]
            self.current_timer = self.load_timer
            self.current_sequencer = 0
            self.start_flag = 1
            self.current_timer_period = self.load_timer
            return
        assert False

    def generate(self) -> npt.NDArray[np.uint8]:
        result = np.zeros(Apu.TICKS_IN_FRAME, dtype=np.uint8)
        if not self.enabled:
            return result
        nesapu.generate_pulse(self, result)
        return result


@dataclass(eq=False)
class Triangle:
    enabled: bool = False
    length_counter_halt: bool = False
    load_timer: int = 0
    load_counter: int = 0
    load_length_counter: int = 0

    # Internal state
    current_tick: int = 0
    current_value: int = 0
    current_timer: int = 0
    length_counter: int = 0
    current_counter: int = 0
    current_sequencer: int = 0
    counter_reload_flag: int = 0

    def set_enabled(self, value: bool) -> None:
        self.enabled = value
        if not value:
            self.length_counter = 0

    def write_register(self, register: ApuRegister, value: int) -> None:
        if register == register.TRIANGLE_CONFIG:
            self.length_counter_halt = bool(value & 0x80)
            self.load_counter = value & 0x7F
            return
        if register == register.TRIANGLE_UNUSED:
            return
        if register == register.TRIANGLE_TIMER:
            self.load_timer &= ~0xFF
            self.load_timer |= value
            return
        if register == register.TRIANGLE_LENGTH_COUNTER:
            self.load_timer &= ~0x700
            self.load_timer |= (value & 0x7) << 8
            self.load_length_counter = value >> 3
            # Set length counter
            if self.enabled:
                self.length_counter = APU_LENGTH_TABLE[self.load_length_counter]
            # Reset internal state
            self.counter_reload_flag = 1
            return
        assert False

    def generate(self) -> npt.NDArray[np.uint8]:
        result = np.zeros(Apu.TICKS_IN_FRAME, dtype=np.uint8)
        nesapu.generate_triangle(self, result)
        return result


@dataclass(eq=False)
class Noise:
    enabled: bool = False

    length_counter_halt: bool = False
    constant_volume: bool = False
    volume: int = 0
    noise_mode: bool = False
    noise_period: int = 0
    load_length_counter: int = 0

    # Internal state
    start_flag: int = 0
    current_tick: int = 0
    current_timer: int = 0
    length_counter: int = 0
    shift_register: int = 1
    divider_period: int = 0
    decay_level_counter: int = 0

    PERIOD_TABLE = [
        4,
        8,
        16,
        32,
        64,
        96,
        128,
        160,
        202,
        254,
        380,
        508,
        762,
        1016,
        2034,
        4068,
    ]

    def set_enabled(self, value: bool) -> None:
        self.enabled = value
        if not value:
            self.length_counter = 0

    def write_register(self, register: ApuRegister, value: int) -> None:
        if register == register.NOISE_CONFIG:
            self.length_counter_halt = bool(value & 0x20)
            self.constant_volume = bool(value & 0x10)
            self.volume = value & 0xF
            return
        if register == register.NOISE_UNUSED:
            return
        if register == register.NOISE_PERIOD:
            self.noise_mode = bool(value & 0x80)
            self.noise_period = self.PERIOD_TABLE[value & 0xF]
            return
        if register == register.NOISE_LENGTH_COUNTER:
            self.load_length_counter = value >> 3
            # Set length counter
            if self.enabled:
                self.length_counter = APU_LENGTH_TABLE[self.load_length_counter]
            # Reset internal state
            self.start_flag = 1
            return
        assert False

    def generate(self) -> npt.NDArray[np.uint8]:
        result = np.zeros(Apu.TICKS_IN_FRAME, dtype=np.uint8)
        nesapu.generate_noise(self, result)
        return result


@dataclass(eq=False)
class Apu:
    frame_counter_mode: int = 0

    pulse1: Pulse = field(default_factory=lambda: Pulse(1))
    pulse2: Pulse = field(default_factory=lambda: Pulse(2))
    triangle: Triangle = field(default_factory=Triangle)
    noise: Noise = field(default_factory=Noise)
    dmc_enabled: bool = False

    # Filter configuration
    filter1_enabled: bool = True
    filter1_cutoff: float = 90.0
    filter2_enabled: bool = True
    filter2_cutoff: float = 442.0
    filter3_enabled: bool = True
    filter3_cutoff: float = 14000.0

    # Filter values
    filter1_previous_in: float = 0
    filter1_previous_out: float = 0
    filter2_previous_in: float = 0
    filter2_previous_out: float = 0
    filter3_previous_in: float = 0
    filter3_previous_out: float = 0

    TICKS_IN_FRAME = 14890

    def write_register(self, cpu: Cpu, register: int, value: int) -> None:
        register = ApuRegister(register)
        if register == register.FRAME_COUNTER:
            self.frame_counter_mode = value >> 7
            return
        if register == register.STATUS:
            self.dmc_enabled = bool(value & 0x10)
            self.noise.set_enabled(bool(value & 0x08))
            self.triangle.set_enabled(bool(value & 0x04))
            self.pulse2.set_enabled(bool(value & 0x02))
            self.pulse1.set_enabled(bool(value & 0x01))
            return
        if register in (
            register.PULSE1_CONFIG,
            register.PULSE1_SWEEP,
            register.PULSE1_TIMER,
            register.PULSE1_LENGTH_COUNTER,
        ):
            self.pulse1.write_register(register, value)
            return
        if register in (
            register.PULSE2_CONFIG,
            register.PULSE2_SWEEP,
            register.PULSE2_TIMER,
            register.PULSE2_LENGTH_COUNTER,
        ):
            self.pulse2.write_register(register, value)
            return
        if register in (
            register.TRIANGLE_CONFIG,
            register.TRIANGLE_UNUSED,
            register.TRIANGLE_TIMER,
            register.TRIANGLE_LENGTH_COUNTER,
        ):
            self.triangle.write_register(register, value)
            return
        if register in (
            register.NOISE_CONFIG,
            register.NOISE_UNUSED,
            register.NOISE_PERIOD,
            register.NOISE_LENGTH_COUNTER,
        ):
            self.noise.write_register(register, value)
            return
        if register in (
            register.DMC_CONFIG,
            register.DMC_LOAD_COUNTER,
            register.DMC_SAMPLE_ADDRESS,
            register.DMC_SAMPLE_LENGTH,
        ):
            return
        assert False

    def generate_dmc(self) -> npt.NDArray[np.uint8]:
        result = np.zeros(Apu.TICKS_IN_FRAME, dtype=np.uint8)
        return result

    def generate(self, audio: npt.NDArray[np.int16]) -> None:
        # self.pulse1.set_enabled(False)
        # self.pulse2.set_enabled(False)
        # self.triangle.set_enabled(False)
        # self.noise.set_enabled(False)

        pulse1 = self.pulse1.generate()
        pulse2 = self.pulse2.generate()
        triangle = self.triangle.generate()
        noise = self.noise.generate()
        dmc = self.generate_dmc()
        nesapu.apu_mixer(self, pulse1, pulse2, triangle, noise, dmc, audio)


class PpuRegister(IntEnum):
    PPUCTRL = 0
    PPUMASK = 1
    PPUSTATUS = 2
    OAMADDR = 3
    OAMDATA = 4
    PPUSCROLL = 5
    PPUADDR = 6
    PPUDATA = 7


@dataclass(eq=False)
class Ppu:
    cartridge: Cartridge
    oam: bytearray = field(default_factory=lambda: bytearray(256))
    ram: bytearray = field(default_factory=lambda: bytearray(8 * 256))
    palette: bytearray = field(default_factory=lambda: bytearray(32))

    # Registers
    ctrl: int = 0
    mask: int = 0
    status: int = 0

    x_scroll: int = 0
    y_scroll: int = 0
    scroll_toggle: int = 0

    oam_addr: int = 0
    ppu_addr: int = 0
    ppu_addr_toggle: int = 0
    delayed_read: int = 0

    vblank: bool = True
    sprite_zero_hit: bool = False
    x_scroll_before_sprite_zero_hit: int = 0
    y_scroll_before_sprite_zero_hit: int = 0

    # Tracking
    instruction_count_at_last_ppu_status_read: int = 0

    # Changes
    background_pattern_table_address_changed: bool = False
    background_tile_changed: set[tuple[int, int]] = field(default_factory=set)
    background_tiles: npt.NDArray[np.uint32] = field(
        default_factory=lambda: np.zeros((240 * 2, 256 * 2), dtype=np.uint32)
    )
    background_tiles_with_palette: list[set[tuple[int, int]]] = field(
        default_factory=lambda: [set(), set(), set(), set()]
    )

    # Properties from PPUCTRL

    @property
    def background_pattern_table_address(self) -> int:
        return 0x1000 if (self.ctrl & 0x10) else 0x0000

    @property
    def sprite_pattern_table_address(self) -> int:
        return 0x1000 if (self.ctrl & 0x08) else 0x0000

    @property
    def sprite_size(self) -> tuple[int, int]:
        return (8, 16) if (self.ctrl & 0x20) else (8, 8)

    @property
    def ram_address_increment(self) -> int:
        return 32 if (self.ctrl & 0x04) else 1

    # Properties from PPUMASK

    @property
    def show_background(self) -> bool:
        return bool(self.mask & 0x08)

    @property
    def show_sprites(self) -> bool:
        return bool(self.mask & 0x10)

    def new_vblank(self) -> None:
        self.x_scroll = 0
        self.y_scroll = 0
        self.scroll_toggle = 0
        self.oam_addr = 0
        self.ppu_addr = 0
        self.ppu_addr_toggle = 0
        self.vblank = True
        self.sprite_zero_hit = False
        self.x_scroll_before_sprite_zero_hit = 0
        self.y_scroll_before_sprite_zero_hit = 0
        self.instruction_count_at_last_ppu_status_read = 0

        self.background_tile_changed.clear()
        self.background_palette_changed = False
        self.background_pattern_table_address_changed = False

    def read_register(self, cpu: Cpu, reg: int) -> int:
        if reg == PpuRegister.PPUCTRL:
            return self.ctrl
        if reg == PpuRegister.PPUMASK:
            return self.mask
        if reg == PpuRegister.PPUSTATUS:
            # Clear
            self.ppu_addr = 0
            self.scroll_toggle = 0
            # Tight loop detected
            if (
                cpu.instruction_count
                <= self.instruction_count_at_last_ppu_status_read + 3
            ):
                if not self.sprite_zero_hit:
                    self.x_scroll_before_sprite_zero_hit = self.x_scroll | (
                        (self.ctrl & 0x01) << 8
                    )
                    self.y_scroll_before_sprite_zero_hit = self.y_scroll | (
                        (self.ctrl & 0x02) << 7
                    )
                    self.sprite_zero_hit = True
                else:
                    self.sprite_zero_hit = False
                    self.vblank = True
            self.instruction_count_at_last_ppu_status_read = cpu.instruction_count
            # First read after VBlank
            if self.vblank:
                self.vblank = False
                return 0x80
            # Sprite 0 Hit has not been reached
            if not self.sprite_zero_hit:
                return 0x00
            # Sprite 0 Hit has been reached
            return 0x40
        if reg == PpuRegister.OAMADDR:
            return self.oam_addr
        if reg == PpuRegister.OAMDATA:
            return self.oam[self.oam_addr]
        if reg == PpuRegister.PPUSCROLL:
            return 0
        if reg == PpuRegister.PPUADDR:
            return 0
        if reg == PpuRegister.PPUDATA:
            result = self.ppu_read(self.ppu_addr)
            self.ppu_addr += self.ram_address_increment
            return result
        assert False

    def write_register(self, cpu: "Cpu", reg: int, value: int) -> None:
        if reg == PpuRegister.PPUCTRL:
            old_address = self.background_pattern_table_address
            self.ctrl = value
            new_address = self.background_pattern_table_address
            if old_address != new_address:
                self.background_pattern_table_address_changed = True
            return
        if reg == PpuRegister.PPUMASK:
            self.mask = value
            return
        if reg == PpuRegister.PPUSTATUS:
            return
        if reg == PpuRegister.OAMADDR:
            self.oam_addr = value
            return
        if reg == PpuRegister.OAMDATA:
            self.oam[self.oam_addr] = value
            return
        if reg == PpuRegister.PPUSCROLL:
            if self.scroll_toggle == 0:
                self.x_scroll = value
            else:
                self.y_scroll = value
            self.scroll_toggle ^= 1
            return
        if reg == PpuRegister.PPUADDR:
            if self.ppu_addr_toggle == 0:
                self.ppu_addr = value << 8
            else:
                self.ppu_addr |= value
            self.ppu_addr_toggle ^= 1
            return
        if reg == PpuRegister.PPUDATA:
            self.ppu_write(self.ppu_addr, value)
            self.ppu_addr += self.ram_address_increment
            return
        assert False, reg

    def write_oam(self, data: bytes | bytearray | memoryview) -> None:
        assert len(data) == 256
        self.oam[:] = data

    def mirror_nametable_addr(self, addr: int) -> int:
        mirrored = (addr - 0x2000) & 0x0FFF
        table = mirrored >> 10
        offset = mirrored & 0x03FF
        mirroring = self.cartridge.mirroring
        if mirroring == "H":
            table = 0 if table in (0, 1) else 1
        elif mirroring == "V":
            table = 0 if table in (0, 2) else 1
        elif mirroring == "1L":
            table = 0
        elif mirroring == "1H":
            table = 1
        else:
            raise ValueError(f"Invalid nametable mirroring mode: {mirroring}")
        return (table << 10) | offset

    def ppu_read(self, addr: int) -> int:
        addr &= 0x3FFF
        # CHR rom access
        if 0x0000 <= addr < 0x2000:
            result, self.delayed_read = self.delayed_read, self.cartridge.ppu_read(addr)
            return result
        if 0x3000 <= addr < 0x3F00:
            addr -= 0x1000
        # Ram access
        if 0x2000 <= addr < 0x3000:
            addr = self.mirror_nametable_addr(addr)
            result, self.delayed_read = self.delayed_read, self.ram[addr]
            return result
        # Palette access
        if 0x3F00 <= addr < 0x4000:
            return self.palette[addr & 0x1F]
        raise ValueError(f"Invalid PPU read: 0x{addr:04x}")

    def ppu_write(self, addr: int, value: int) -> None:
        addr &= 0x3FFF
        # Ram access
        if 0x3000 <= addr < 0x3F00:
            addr -= 0x1000
        if 0x2000 <= addr < 0x3000:
            logical_addr = (addr - 0x2000) & 0x0FFF
            physical_addr = self.mirror_nametable_addr(addr)
            if self.ram[physical_addr] != value:
                self.background_tile_changed.update(self.addr_to_indexes(logical_addr))
            self.ram[physical_addr] = value
            return
        # Palette access
        if 0x3F00 <= addr < 0x4000:
            addr &= 0x1F
            if addr in (0x00, 0x04, 0x08, 0x0C):
                self.palette[addr | 0x10] = value
            elif addr in (0x10, 0x14, 0x18, 0x1C):
                self.palette[addr & ~0x10] = value
            elif addr < 0x10 and self.palette[addr] != value:
                self.background_tile_changed.update(
                    self.background_tiles_with_palette[addr >> 2]
                )
            self.palette[addr] = value
            return
        # CHR RAM access
        if 0x0000 <= addr < 0x2000:
            self.cartridge.ppu_write(addr, value)
            self.render_tile.cache_clear()
            return
        raise ValueError(f"Invalid PPU write: 0x{addr:04x}")

    def render(self, video: npt.NDArray[np.uint32]) -> None:
        self.render_background_color(video)
        self.render_sprite(video, behind=True)
        self.render_background(video)
        self.render_sprite(video, behind=False)

    def render_background_color(self, video: npt.NDArray[np.uint32]) -> None:
        background_color = self.palette[0]
        video.fill(nesppu.get_color(background_color))

    def update_tiles(self) -> None:
        base_pattern_address = self.background_pattern_table_address
        # Draw everything
        if (
            self.background_pattern_table_address_changed
            or self.background_palette_changed
        ):
            for x_index in range(64):
                for y_index in range(30):
                    self.update_tile(y_index, x_index, base_pattern_address)
            return
        # Draw changes
        for y_index, x_index in self.background_tile_changed:
            self.update_tile(y_index, x_index, base_pattern_address)

    def index_to_addr(self, y: int, x: int) -> tuple[int, int]:
        nametable = ((y & 0x20) << 6) | ((x & 0x20) << 5)
        pattern = nametable | ((y & 0b00011111) << 5) | (x & 0b00011111)
        palette = nametable | 0x03C0
        palette |= (y & 0b00011100) << 1
        palette |= (x & 0b00011100) >> 2
        return pattern, palette

    def addr_to_indexes(self, addr: int) -> list[tuple[int, int]]:
        y = ((addr >> 11) & 0x01) << 5
        x = ((addr >> 10) & 0x01) << 5
        addr &= 0x3FF
        if addr < 0x03C0:
            y |= (addr >> 5) & 0x1F
            x |= (addr >> 0) & 0x1F
            return [(y, x)]
        y |= (addr & 0b00111000) >> 1
        x |= (addr & 0b00000111) << 2
        return [(y | dy, x | dx) for dy in range(4) for dx in range(4)]

    def update_tile(
        self, y_index: int, x_index: int, base_pattern_address: int
    ) -> None:
        # Filter
        if y_index in (30, 31, 62, 63):
            return
        # Get nametable
        pattern_ram_address, palette_ram_address = self.index_to_addr(y_index, x_index)
        pattern_ram_address = self.mirror_nametable_addr(0x2000 + pattern_ram_address)
        palette_ram_address = self.mirror_nametable_addr(0x2000 + palette_ram_address)
        # Get pattern address
        pattern_address = self.ram[pattern_ram_address]
        pattern_address = (pattern_address << 4) | base_pattern_address
        # Get colors
        palette_address = self.ram[palette_ram_address]
        shift = ((y_index & 0x2) << 1) | (x_index & 0x02)
        palette_address = ((palette_address >> shift) & 0x3) << 2
        colors = self.palette[palette_address + 1 : palette_address + 4]
        # Update palette info
        palette_address >>= 2
        entry = (y_index, x_index)
        if entry not in self.background_tiles_with_palette[palette_address]:
            for tile_set in self.background_tiles_with_palette:
                tile_set.discard(entry)
            self.background_tiles_with_palette[palette_address].add(entry)
        # Get tile
        tile = self.render_tile(
            pattern_address, bytes(colors), self.cartridge.chr_cache_key
        )
        # Blit tile
        y_pixel = y_index << 3
        x_pixel = x_index << 3
        if y_index >= 32:
            y_pixel -= 16
        self.background_tiles[y_pixel : y_pixel + 8, x_pixel : x_pixel + 8] = tile

    def render_background(self, video: npt.NDArray[np.uint32]) -> None:
        self.update_tiles()
        if not self.show_background:
            return
        first_row = 8  # Hide first and last row like most monitors
        sprite_zero_hit_y = self.oam[0] + 8
        x_scroll = self.x_scroll | ((self.ctrl & 0x01) << 8)
        nesppu.blit(
            self.background_tiles[:sprite_zero_hit_y, :], video, (-first_row, 0)
        )
        nesppu.blit(
            self.background_tiles[sprite_zero_hit_y:, :],
            video,
            (sprite_zero_hit_y - first_row, 0 - x_scroll),
        )
        nesppu.blit(
            self.background_tiles[sprite_zero_hit_y:, :],
            video,
            (sprite_zero_hit_y - first_row, 512 - x_scroll),
        )

    def render_sprite(
        self, video: npt.NDArray[np.uint32], behind: bool = False
    ) -> None:
        if not self.show_sprites:
            return
        palette = self.palette
        pattern_table_address = self.sprite_pattern_table_address
        first_row = 8  # Hide first and last row like most monitors
        for i in reversed(range(64)):
            y, index, attr, x = self.oam[i * 4 : i * 4 + 4]
            # Below the screen
            if y >= 240:
                continue
            # Behind or in front of the background
            if behind != bool(attr & 0x20):
                continue
            # Pattern address
            pattern_addr = (index << 4) | pattern_table_address
            # Colors
            color_index = attr & 0x03
            palette_addr = 0x10 | (color_index << 2)
            colors = palette[palette_addr + 1 : palette_addr + 4]
            # Tile
            if self.sprite_size == (8, 8):
                tile = self.render_tile(
                    pattern_addr, bytes(colors), self.cartridge.chr_cache_key
                )
            else:
                pattern_table_address = (index & 0x01) << 12
                index &= 0xFE
                top = self.render_tile(
                    (index << 4) | pattern_table_address,
                    bytes(colors),
                    self.cartridge.chr_cache_key,
                )
                bottom = self.render_tile(
                    ((index + 1) << 4) | pattern_table_address,
                    bytes(colors),
                    self.cartridge.chr_cache_key,
                )
                tile = np.vstack((top, bottom))
            # Vertical flip
            if attr & 0x80:
                tile = tile[::-1, :]
            # Horizontal flip
            if attr & 0x40:
                tile = tile[:, ::-1]
            # Blit
            nesppu.blit(tile, video, (y - first_row, x))

    @lru_cache(maxsize=None)
    def render_tile(
        self, pattern_addr: int, colors: bytes, chr_cache_key: tuple[int, int, int, int]
    ) -> npt.NDArray[np.uint32]:
        result = np.zeros((8, 8), dtype=np.uint32)
        tile_data = bytes(self.cartridge.ppu_read(pattern_addr + i) for i in range(16))
        nesppu.render_tile(tile_data, 0, colors, result)
        return result


@dataclass(eq=False)
class Cpu:
    cartridge: Cartridge
    ppu: Ppu
    apu: Apu
    ram: bytearray = field(default_factory=lambda: bytearray(8 * 256))

    # Internal registers
    pc: int = 0
    sp: int = 0
    a: int = 0
    x: int = 0
    y: int = 0

    # Flags
    n: bool = False
    z: bool = False
    c: bool = False
    v: bool = False
    i: bool = False
    d: bool = False

    # Tracking
    frame: int = 0
    instruction_count: int = 0
    max_instructions_per_run: int = 200_000

    # IO
    input_value: int = 0

    @property
    def rom(self) -> bytes:
        return self.cartridge.mapped_prg_rom

    # CPU Bus access

    def cpu_read(self, addr: int) -> int:
        # Ram access
        if 0 <= addr < 0x0800:
            return self.ram[addr]
        # Mirror ram access
        if 0x0800 <= addr < 0x2000:
            return self.ram[addr & 0x07FF]
        # Rom access
        if 0x8000 <= addr < 0x10000:
            return self.cartridge.cpu_read(addr)
        # PPU access
        if 0x2000 <= addr < 0x4000:
            return self.ppu.read_register(self, addr & 0x7)
        # APU/IO access
        if 0x4000 <= addr < 0x4014:
            return 0
        # Sound channel status
        if addr == 0x4015:
            return (
                (self.apu.pulse1.length_counter > 0)
                | ((self.apu.pulse2.length_counter > 0) << 1)
                | ((self.apu.triangle.length_counter > 0) << 2)
                | ((self.apu.noise.length_counter > 0) << 3)
                | (self.apu.dmc_enabled << 4)
            )
        # Joystick 1 data
        if addr == 0x4016:
            result = self.input_value & 0x01
            self.input_value >>= 1
            return result
        # Joystick 2 data
        if addr == 0x4017:
            return 0
        # Invalid access
        if 0x6000 <= addr < 0x8000:
            return self.cartridge.cpu_read(addr)
        raise ValueError(f"Invalid read access: 0x{addr:04x} (pc=0x{self.pc:04x})")

    def cpu_write(self, addr: int, value: int) -> None:
        # Ram access
        if 0 <= addr < 0x0800:
            self.ram[addr] = value
            return
        # Mirror ram access
        if 0x0800 <= addr < 0x2000:
            self.ram[addr & 0x07FF] = value
            return
        # PPU access
        if 0x2000 <= addr < 0x4000:
            return self.ppu.write_register(self, addr & 0x7, value)
        # APU access
        if 0x4000 <= addr < 0x4014:
            self.apu.write_register(self, addr & 0x1F, value)
            return
        # OAM DMA
        if addr == 0x4014:
            start = value << 8
            stop = (value + 1) << 8
            data = self.ram[start:stop]
            return self.ppu.write_oam(data)
        # Sound channel control
        if addr == 0x4015:
            self.apu.write_register(self, addr & 0x1F, value)
            return
        # Joystick 1 data
        if addr == 0x4016:
            return
        # Joystick 2 data
        if addr == 0x4017:
            self.apu.write_register(self, addr & 0x1F, value)
            return
        # Rom access
        if 0x6000 <= addr < 0x10000:
            self.cartridge.cpu_write(addr, value)
            return
        raise ValueError(f"Invalid write access: 0x{addr:04x} (pc=0x{self.pc:04x})")

    # Entry points

    def load_nmi_entrypoint(self) -> None:
        self.frame += 1
        self.pc = self.cpu_read(0xFFFA)
        self.pc |= self.cpu_read(0xFFFB) << 8

    def load_rst_entrypoint(self) -> None:
        self.pc = self.cpu_read(0xFFFC)
        self.pc |= self.cpu_read(0xFFFD) << 8

    # Run CPU instructions

    def run_instructions(self) -> None:
        jmp = 0x4C
        rti = 0x40
        budget = 0x100
        opc = nescpu.run(self)
        if opc == budget:
            raise InstructionBudgetExhausted()
        # Slow run
        if opc == jmp:
            raise InfiniteLoop()
        assert opc == rti
        return


def parse_ines(source: str) -> Cartridge:

    with open(source, "rb") as input_file:
        header = input_file.read(16)
        if header[:4] != b"NES\x1a":
            raise ValueError(f"{source} is not an iNES ROM")
        prg_rom_size = header[4] * 16 * 1024
        chr_rom_size = header[5] * 8 * 1024
        flag6, flag7, flag8, flag9, flag10 = header[6:11]

        mapper = (flag6 >> 4) | (flag7 & 0xF0)
        mirroring = "V" if bool(flag6 & 0x01) else "H"
        cartridge_has_prg_ram = bool(flag6 & 0x02)
        has_trainer = bool(flag6 & 0x04)
        ignore_mirroring_control = bool(flag6 & 0x08)

        trainer = input_file.read(512) if has_trainer else None

        prg_rom = input_file.read(prg_rom_size)
        chr_rom = input_file.read(chr_rom_size)

        if len(prg_rom) != prg_rom_size or len(chr_rom) != chr_rom_size:
            raise ValueError(f"{source} is truncated")

    if mapper not in (0, 1):
        raise UnsupportedCartridgeError(
            f"Unsupported NES mapper {mapper}; only mapper 0 (NROM) and mapper 1 (MMC1) are supported"
        )
    if mapper == 0:
        if len(prg_rom) == 16 * 1024:
            prg_rom *= 2
        elif len(prg_rom) != 32 * 1024:
            raise UnsupportedCartridgeError(
                f"Unsupported mapper 0 PRG ROM size: {len(prg_rom)} bytes"
            )
        if len(chr_rom) not in (0, 8 * 1024):
            raise UnsupportedCartridgeError(
                f"Unsupported mapper 0 CHR ROM size: {len(chr_rom)} bytes"
            )
    elif mapper == 1:
        if not prg_rom or len(prg_rom) % (16 * 1024):
            raise UnsupportedCartridgeError(
                f"Unsupported mapper 1 PRG ROM size: {len(prg_rom)} bytes"
            )
        if len(chr_rom) % (8 * 1024):
            raise UnsupportedCartridgeError(
                f"Unsupported mapper 1 CHR ROM size: {len(chr_rom)} bytes"
            )
    return Cartridge(
        mapper,
        mirroring,
        cartridge_has_prg_ram,
        has_trainer,
        ignore_mirroring_control,
        trainer,
        prg_rom,
        chr_rom,
    )


class Nes(Console):
    WIDTH = 256
    HEIGHT = 240 - 16
    FPS = 60
    TICKS_IN_FRAME = 29780

    INPUT_MAP = {
        Console.Input.A: 0x01,
        Console.Input.B: 0x02,
        Console.Input.SELECT: 0x04,
        Console.Input.START: 0x08,
        Console.Input.UP: 0x10,
        Console.Input.DOWN: 0x20,
        Console.Input.LEFT: 0x40,
        Console.Input.RIGHT: 0x80,
    }

    @classmethod
    def add_console_arguments(cls, parser: ArgumentParser) -> None:
        pass

    @classmethod
    def pop_console_arguments(cls, namespace: Namespace) -> Callable[[], Console]:
        romfile: Path = namespace.romfile
        return lambda: cls(romfile)

    def __init__(self, romfile: Path) -> None:
        self.current_state = 0
        super().__init__(romfile)
        self.cartridge = parse_ines(self.romfile)
        self.cpu = Cpu(
            self.cartridge,
            Ppu(self.cartridge),
            Apu(),
        )
        # Run RST
        self.cpu.load_rst_entrypoint()
        try:
            self.cpu.run_instructions()
        except (InfiniteLoop, InstructionBudgetExhausted):
            pass

    @property
    def apu(self) -> Apu:
        return self.cpu.apu

    @property
    def ppu(self) -> Ppu:
        return self.cpu.ppu

    def set_input(self, input_set: set[Console.Input]) -> None:
        value = sum(self.INPUT_MAP.get(key, 0) for key in input_set)
        self.cpu.input_value = value

    def advance_one_frame(
        self, video: npt.NDArray[np.uint32], audio: npt.NDArray[np.int16]
    ) -> tuple[bool, int]:
        self.ppu.new_vblank()
        self.cpu.load_nmi_entrypoint()
        try:
            self.cpu.run_instructions()
        except (InfiniteLoop, InstructionBudgetExhausted):
            pass
        self.ppu.render(video)
        self.apu.generate(audio)
        return True, self.TICKS_IN_FRAME

    def set_current_state(self, state: int) -> None:
        self.current_state = state % 10

    def get_current_state(self) -> int:
        return self.current_state

    def load_state(self) -> None:
        path = f"{self.romfile}.{self.current_state}.state"
        try:
            with open(path, "rb") as f:
                cpu: Cpu = pickle.loads(zlib.decompress(f.read()))
        except OSError:
            return
        cpu.cartridge = self.cartridge
        cpu.ppu.cartridge = self.cartridge
        self.cpu = cpu

    def save_state(self) -> None:
        path = f"{self.romfile}.{self.current_state}.state"
        cpu = deepcopy(self.cpu)
        del cpu.cartridge
        del cpu.ppu.cartridge
        with open(path, "wb") as f:
            f.write(zlib.compress(pickle.dumps(cpu)))


def main(parser_args: tuple[str, ...] | None = None) -> None:
    try:
        gambaterm_main(parser_args, console_cls=Nes)
    except (UnsupportedCartridgeError, ValueError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
