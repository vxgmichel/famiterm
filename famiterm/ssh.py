from __future__ import annotations

from .run import Nes
from gambaterm.ssh import main as gambaterm_ssh_main


def main(parser_args: tuple[str, ...] | None = None) -> None:
    gambaterm_ssh_main(parser_args, console_cls=Nes)
