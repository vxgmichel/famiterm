from __future__ import annotations

from .run import Nes
from gambaterm.telnet import main as gambaterm_telnet_main


def main(parser_args: tuple[str, ...] | None = None) -> None:
    gambaterm_telnet_main(parser_args, console_cls=Nes)
