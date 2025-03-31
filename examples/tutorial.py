"""Similar to the MineRL "Hello World" tutorial.
Just do random steps and display the results"""

import argparse
import pprint
import sys

import mcio_remote as mcio
from mcio_remote.envs import mcio_env

# import gymnasium as gym


def tutorial(steps: int, instance_name: str | None, world_name: str | None) -> None:
    if instance_name is not None:
        if world_name is None:
            raise ValueError("World name must be provided if instance name is provided")
        opts = mcio.types.RunOptions.for_launch(instance_name, world_name)
    else:
        opts = mcio.types.RunOptions.for_connect()

    # gym.make() works, but I prefer just creating the env instance directly.
    # env = gym.make("mcio_env/MCioEnv-v0", render_mode="human", run_options=opts)
    env = mcio_env.MCioEnv(opts)

    if steps == 0:
        steps = sys.maxsize  # Go forever
    step = 0
    setup_commands = [
        "time set day",
        # "teleport @s 14 135 140 180 0",
        "summon minecraft:sheep ~2 ~2 ~2",
        "summon minecraft:cow ~-2 ~2 ~-2",
    ]
    observation, info = env.reset(options={"commands": setup_commands})
    print_step(step, None, observation)
    step += 1
    done = False

    while not done and step < steps:
        action = env.action_space.sample()

        # Cycle jumping on and off
        cycle = (steps // 50) % 2
        if cycle == 0:
            action["SPACE"] = mcio_env.PRESS
        elif cycle == 1:
            action["SPACE"] = mcio_env.NO_PRESS

        # Limit some actions
        action["cursor_delta"] = action["cursor_delta"].clip(-20, 20)
        action["E"] = mcio_env.NO_PRESS
        action["S"] = mcio_env.NO_PRESS

        # Go forward and press attack button
        action["W"] = mcio_env.PRESS
        action["BUTTON_LEFT"] = mcio_env.PRESS
        observation, reward, terminated, truncated, info = env.step(action)
        print_step(step, action, observation)
        step += 1
        done = terminated or truncated

    env.close()


def print_step(
    step: int,
    action: mcio_env.MCioAction | None = None,
    observation: mcio_env.MCioObservation | None = None,
) -> None:
    print(f"Step {step}:")
    if action is not None:
        print(f"Action:\n{pprint.pformat(action)}")
    if observation is not None:
        print(f"Obs:\n{obs_to_string(observation)}")
    print("-" * 10)


def obs_to_string(obs: mcio_env.MCioObservation) -> str:
    """Return a pretty version of the observation as a string.
    Prints the shape of the frame rather than the frame itself"""
    frame = obs["frame"]
    obs["frame"] = frame.shape
    formatted = pprint.pformat(obs)
    obs["frame"] = frame
    return formatted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate actions and observations")

    mcio.util.logging_add_arg(parser)

    parser.add_argument(
        "--steps", "-s", type=int, default=100, help="Number of steps, 0 for forever"
    )
    parser.add_argument(
        "--instance-name",
        "-i",
        type=str,
        help="Name of the Minecraft instance to launch",
    )
    parser.add_argument("--world", "-w", type=str, help="World name")

    args = parser.parse_args()
    mcio.util.logging_init(args=args)
    return args


if __name__ == "__main__":
    args = parse_args()

    tutorial(args.steps, args.instance_name, args.world)
