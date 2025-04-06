"""Just send F3"""

import argparse

import mcio_remote as mcio
from mcio_remote.envs import mcio_env


def toggle() -> None:
    opts = mcio.types.RunOptions.for_connect()
    env = mcio_env.MCioEnv(opts)

    env.reset()
    env.toggle_f3()
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send F3")
    mcio.util.logging_add_arg(parser)
    args = parser.parse_args()
    mcio.util.logging_init(args=args)
    return args


if __name__ == "__main__":
    args = parse_args()
    toggle()
