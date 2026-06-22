from __future__ import annotations


def main(parser_args: tuple[str, ...] | None = None) -> None:
    from .run import main as run_main

    run_main(parser_args)

__all__ = ["main"]
