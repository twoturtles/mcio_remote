"""Demonstrates command delay and out-of-order"""

import argparse
import time

import mcio_remote as mcio
from mcio_remote.envs import mcio_env


def setup() -> None:
    opts = mcio.types.RunOptions.for_connect()
    env = mcio_env.MCioEnv(opts, render_mode="human")

    # Instead of the expected result, you get:
    # Look down ... day ... look up ... night
    # Should be day, down, night, up
    env.reset()
    skipn(env, 20)
    print("Set day")
    cmds(env, ["time set day"])
    cmds(env, ["teleport @s ~ ~ ~ 0 45"])  # Look down
    skipn(env, 25)
    # skipx(env)
    print("Pause")
    time.sleep(1)
    print("Set night")
    cmds(env, ["time set night"])
    cmds(env, ["teleport @s ~ ~ ~ 0 -45"])  # Look up
    skipn(env, 25)
    # skipx(env)
    env.close()


def cmds(env: mcio_env.MCioEnv, commands: list[str]) -> None:
    env.step(env.get_noop_action(), options={"commands": commands})


def skipn(env: mcio_env.MCioEnv, steps: int) -> None:
    for i in range(steps):
        observation, reward, terminated, truncated, info = env.step(
            env.get_noop_action()
        )
        time.sleep(0.1)
        print(f"Skip {i+1}")


def skipx(env: mcio_env.MCioEnv) -> None:
    done = False
    i = 0
    while not done:
        observation, reward, terminated, truncated, info = env.step(
            env.get_noop_action()
        )
        i += 1
        key = input(f"{i}: Step? ")
        if key.lower() == "n":
            done = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test")

    mcio.util.logging_add_arg(parser)

    # parser.add_argument(
    #     "mode",
    #     metavar="mode",
    #     type=str,
    # )

    args = parser.parse_args()
    mcio.util.logging_init(args=args)
    return args


if __name__ == "__main__":
    args = parse_args()
    setup()
