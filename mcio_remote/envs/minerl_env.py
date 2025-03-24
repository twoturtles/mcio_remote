"""
This provides an environment compatible with the minerl 1.0 action and observation spaces.
"""

from dataclasses import dataclass
from typing import Any

import glfw  # type: ignore
import numpy as np
from gymnasium import spaces

import mcio_remote as mcio
from mcio_remote.types import InputID, InputType

from . import env_util
from .base_env import McioBaseEnv, McioBaseEnvArgs

# Stub in the action and observation space types
type MinerlAction = dict[str, Any]
type MinerlObservation = dict[str, Any]


# Map from Minerl action name to Minecraft input
INPUT_MAP: dict[str, InputID] = {
    "attack": InputID(InputType.MOUSE, glfw.MOUSE_BUTTON_LEFT),
    "use": InputID(InputType.MOUSE, glfw.MOUSE_BUTTON_RIGHT),
    "pickItem": InputID(InputType.MOUSE, glfw.MOUSE_BUTTON_MIDDLE),
    "forward": InputID(InputType.KEY, glfw.KEY_W),
    "left": InputID(InputType.KEY, glfw.KEY_A),
    "right": InputID(InputType.KEY, glfw.KEY_D),
    "back": InputID(InputType.KEY, glfw.KEY_S),
    "drop": InputID(InputType.KEY, glfw.KEY_Q),
    "inventory": InputID(InputType.KEY, glfw.KEY_E),
    "jump": InputID(InputType.KEY, glfw.KEY_SPACE),
    "sneak": InputID(InputType.KEY, glfw.KEY_LEFT_SHIFT),
    "sprint": InputID(InputType.KEY, glfw.KEY_LEFT_CONTROL),
    "swapHands": InputID(InputType.KEY, glfw.KEY_F),
    "hotbar.1": InputID(InputType.KEY, glfw.KEY_1),
    "hotbar.2": InputID(InputType.KEY, glfw.KEY_2),
    "hotbar.3": InputID(InputType.KEY, glfw.KEY_3),
    "hotbar.4": InputID(InputType.KEY, glfw.KEY_4),
    "hotbar.5": InputID(InputType.KEY, glfw.KEY_5),
    "hotbar.6": InputID(InputType.KEY, glfw.KEY_6),
    "hotbar.7": InputID(InputType.KEY, glfw.KEY_7),
    "hotbar.8": InputID(InputType.KEY, glfw.KEY_8),
    "hotbar.9": InputID(InputType.KEY, glfw.KEY_9),
}


@dataclass
class MinerlEnvArgs(McioBaseEnvArgs):
    """See McioBaseEnvArgs for more info"""

    pass


class MinerlEnv(McioBaseEnv[MinerlObservation, MinerlAction]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(self, args: MinerlEnvArgs) -> None:
        """
        Attempt at Minerl 1.0 compatible environment. Replicates the Minerl action and observation spaces.
        """
        super().__init__(args)

        # Used for the ESC action
        self.terminated = False

        self.observation_space = spaces.Dict(
            {
                "pov": spaces.Box(
                    low=0,
                    high=255,
                    # shape=(height, width, channels)
                    shape=(self.run_options.height, self.run_options.width, 3),
                    dtype=np.uint8,
                ),
            }
        )

        _action_space: dict[str, Any] = {
            key: spaces.Discrete(2) for key in INPUT_MAP.keys()
        }
        # ESC is a special case in minerl. It's not passed to Minecraft. Instead
        # it signals the environment to terminate.
        _action_space["ESC"] = spaces.Discrete(2)
        # camera is the change in degrees of (pitch, yaw)
        _action_space["camera"] = spaces.Box(low=-180.0, high=180.0, shape=(2,))
        self.action_space = spaces.Dict(_action_space)

        # Env helpers
        self.input_mgr = env_util.InputStateManager()
        self.cursor_map = env_util.DegreesToPixels()

    def _process_step(
        self, action: MinerlAction, observation: MinerlObservation
    ) -> tuple[int, bool, bool]:
        # reward, terminated, truncated
        return 0, self.terminated, False

    def _packet_to_observation(
        self, packet: mcio.network.ObservationPacket
    ) -> MinerlObservation:
        """Convert an ObservationPacket to the environment observation_space"""
        obs: MinerlObservation = {
            "pov": packet.get_frame_with_cursor(),
        }
        return obs

    def _action_to_packet(
        self, action: MinerlAction, commands: list[str] | None = None
    ) -> mcio.network.ActionPacket:
        """Convert from the environment action_space to an ActionPacket
        Always populate the packet with all possible keys/buttons. Minecraft can
        handle repeated PRESS and RELEASE actions. Not set in the action = RELEASE."""
        packet = mcio.network.ActionPacket()
        packet.inputs = self.input_mgr.process_action(action, INPUT_MAP)
        packet.cursor_pos = [
            self.cursor_map.update(
                pitch_delta=action["camera"][0], yaw_delta=action["camera"][1]
            )
        ]
        packet.commands = commands or []

        if action["ESC"]:
            # Signal termination
            self.terminated = True

        return packet
