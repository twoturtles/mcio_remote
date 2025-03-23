"""
This provides an environment compatible with the minerl 1.0 action and observation spaces.
"""

from typing import Any

import glfw  # type: ignore
import numpy as np
from gymnasium import spaces

import mcio_remote as mcio
from mcio_remote.types import InputID, InputType

from . import env_util
from .base_env import McioBaseEnv

"""
Notes:

Minerl Observation Space:
Dict(pov:Box(low=0, high=255, shape=(360, 640, 3))), dtype = 'uint8'


Minerl Action Space:
Dict({
    "ESC": "Discrete(2)",
    "attack": "Discrete(2)",
    "back": "Discrete(2)",
    "camera": "Box(low=-180.0, high=180.0, shape=(2,))",
    "drop": "Discrete(2)",
    "forward": "Discrete(2)",
    "hotbar.1": "Discrete(2)",
    "hotbar.2": "Discrete(2)",
    "hotbar.3": "Discrete(2)",
    "hotbar.4": "Discrete(2)",
    "hotbar.5": "Discrete(2)",
    "hotbar.6": "Discrete(2)",
    "hotbar.7": "Discrete(2)",
    "hotbar.8": "Discrete(2)",
    "hotbar.9": "Discrete(2)",
    "inventory": "Discrete(2)",
    "jump": "Discrete(2)",
    "left": "Discrete(2)",
    "pickItem": "Discrete(2)", Mouse 2
    "right": "Discrete(2)",
    "sneak": "Discrete(2)",
    "sprint": "Discrete(2)",    Left Control
    "swapHands": "Discrete(2)", F
    "use": "Discrete(2)"
})

Sample action:
OrderedDict([('ESC', array(0)), ('attack', array(1)), ('back', array(0)), ('camera', array([-21.149803,  41.296047], dtype=float32)), ('drop', array(1)), ('forward', array(1)), ('hotbar.1', array(0)), ('hotbar.2', array(1)), ('hotbar.3', array(0)), ('hotbar.4', array(1)), ('hotbar.5', array(1)), ('hotbar.6', array(1)), ('hotbar.7', array(0)), ('hotbar.8', array(1)), ('hotbar.9', array(0)), ('inventory', array(1)), ('jump', array(1)), ('left', array(0)), ('pickItem', array(1)), ('right', array(0)), ('sneak', array(0)), ('sprint', array(0)), ('swapHands', array(1)), ('use', array(1))])

Minerl behavior notes:
    - inventory must be released before it toggles the inventory screen. mcio gui behaves the same,
    so this must be a Minecraft behavior. E.g.:
        - step 1 inventory=1 -- opens inventory
        - step 2 inventory=1 -- inventory stays open
        - step 3 inventory=0 -- inventory stays open
        - step 4 inventory=1 -- inventory closes
    - swap hands has the same behavior

    - sneak=1 to crouch, sneak=0 to stand
"""


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


class MinerlEnv(McioBaseEnv[MinerlObservation, MinerlAction]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Attempt at Minerl 1.0 compatible environment. This only replicates the Minerl
        action and observation spaces.

        See **McioBaseEnv** for docs on parameters
        """
        super().__init__(*args, **kwargs)

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
