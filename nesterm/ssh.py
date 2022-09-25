from .run import Nes
from gambaterm.ssh import main as gambaterm_ssh_main


def main(parser_args=None):
    return gambaterm_ssh_main(console_cls=Nes)
