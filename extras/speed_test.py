"""
Basic step speed test. Drives MCio in sync mode as fast as possible.

mcio inst launch DemoInstance -m sync -w DemoWorld -W 640 -H 360
SPEED-TEST mode=mcio steps=5000 setup=0.19 run=16.92 steps_per_sec=295.50
"""

import argparse
import logging
import time
from collections import defaultdict
from typing import Any

from tqdm import tqdm


def minerl_setup() -> Any:
    # gym version 0.23.1
    import gym  # type: ignore

    # minerl version 1.0.2
    import minerl  # type: ignore # noqa: F401  # needed for gym registration

    # logging.basicConfig(level=logging.DEBUG)
    # Defaults to frame shape (360, 640, 3)
    env = gym.make("MineRLBasaltFindCave-v0")
    env.reset()
    return env


def mcio_setup(render: bool, connect: bool) -> Any:
    import mcio_remote as mcio
    from mcio_remote.envs import mcio_env

    if connect:
        # To launch an instance:
        #  mcio inst launch DemoInstance -m sync -w DemoWorld -W 640 -H 360
        opts = mcio.types.RunOptions.for_connect(width=640, height=360)
    else:
        opts = mcio.types.RunOptions.for_launch(
            "DemoInstance", "DemoWorld", width=640, height=360
        )

    render_mode = "human" if render else None
    env = mcio_env.MCioEnv(opts, render_mode=render_mode)
    env.reset()
    return env


def minerl_run(
    env: Any, num_steps: int, render: bool, steps_completed: list[int]
) -> None:
    action: dict[str, Any] = defaultdict(
        int
    )  # This will return 0 for any unspecified key
    # action["camera"] = [0, 10]
    print(action)
    for _ in tqdm(range(num_steps)):
        env.step(action)
        if render:
            env.render()
        steps_completed[0] += 1


def mcio_run(
    env: Any, num_steps: int, render: bool, steps_completed: list[int]
) -> None:
    from mcio_remote.envs import mcio_env

    assert isinstance(env, mcio_env.MCioEnv)
    action = env.get_noop_action()
    # action["cursor_pos_rel"][:] = [10, 0]
    print(action)
    # flip = 0
    for _ in tqdm(range(num_steps)):
        # action["mouse_buttons"]["RIGHT"] = flip
        # flip = 1 - flip
        env.step(action)
        if render:
            env.render()
        steps_completed[0] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", type=str, choices=["mcio", "minerl"], help="Test mode")
    parser.add_argument("--steps", "-s", type=int, default=1000, help="Number of steps")
    parser.add_argument(
        "--render", "-r", action="store_true", help="render (show output frames)"
    )
    parser.add_argument(
        "--connect",
        "-c",
        action="store_true",
        help="Connect to a separately launched Minecraft (mcio only)",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    start = time.perf_counter()
    if args.mode == "minerl":
        env = minerl_setup()
    else:
        env = mcio_setup(args.render, args.connect)
    setup_time = time.perf_counter() - start

    start = time.perf_counter()
    steps_completed = [0]
    try:
        if args.mode == "minerl":
            minerl_run(env, args.steps, args.render, steps_completed)
        else:
            mcio_run(env, args.steps, args.render, steps_completed)
    except KeyboardInterrupt:
        print("Exiting...")
    run_time = time.perf_counter() - start
    env.close()

    steps = steps_completed[0]
    print(
        f"SPEED-TEST mode={args.mode} steps={steps} setup={setup_time:.2f} "
        f"run={run_time:.2f} steps_per_sec={steps/run_time:.2f}"
    )


if __name__ == "__main__":
    main()
