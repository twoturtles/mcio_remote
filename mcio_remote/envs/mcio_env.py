"""
This is a sample gym environment for MCio
"""

from typing import Any

import glfw  # type: ignore
import numpy as np
from gymnasium import spaces

import mcio_remote as mcio
from mcio_remote.types import InputID, InputType, RunOptions

from . import env_util
from .base_env import MCioBaseEnv

# Stub in the action and observation space types
type MCioAction = dict[str, Any]
type MCioObservation = dict[str, Any]

# key / button states in action spaces
NO_PRESS = np.int64(0)
PRESS = np.int64(1)

# Map from action name to Minecraft input
# "cursor_delta" key is also available, added in __init__()
INPUT_MAP: dict[str, InputID] = {
    "LEFT_BUTTON": InputID(InputType.MOUSE, glfw.MOUSE_BUTTON_LEFT),
    "RIGHT_BUTTON": InputID(InputType.MOUSE, glfw.MOUSE_BUTTON_RIGHT),
    "MIDDLE_BUTTON": InputID(InputType.MOUSE, glfw.MOUSE_BUTTON_MIDDLE),
    "W": InputID(InputType.KEY, glfw.KEY_W),
    "A": InputID(InputType.KEY, glfw.KEY_A),
    "D": InputID(InputType.KEY, glfw.KEY_D),
    "S": InputID(InputType.KEY, glfw.KEY_S),
    "Q": InputID(InputType.KEY, glfw.KEY_Q),
    "E": InputID(InputType.KEY, glfw.KEY_E),
    "SPACE": InputID(InputType.KEY, glfw.KEY_SPACE),
    "LEFT_SHIFT": InputID(InputType.KEY, glfw.KEY_LEFT_SHIFT),
    "LEFT_CONTROL": InputID(InputType.KEY, glfw.KEY_LEFT_CONTROL),
    "F": InputID(InputType.KEY, glfw.KEY_F),
    "1": InputID(InputType.KEY, glfw.KEY_1),
    "2": InputID(InputType.KEY, glfw.KEY_2),
    "3": InputID(InputType.KEY, glfw.KEY_3),
    "4": InputID(InputType.KEY, glfw.KEY_4),
    "5": InputID(InputType.KEY, glfw.KEY_5),
    "6": InputID(InputType.KEY, glfw.KEY_6),
    "7": InputID(InputType.KEY, glfw.KEY_7),
    "8": InputID(InputType.KEY, glfw.KEY_8),
    "9": InputID(InputType.KEY, glfw.KEY_9),
}

# Mouse movement in pixels relative to the current position
CURSOR_DELTA_ZERO = np.array((0.0, 0.0), dtype=np.int32)
CURSOR_DELTA_ZERO.flags.writeable = False


class MCioEnv(MCioBaseEnv[MCioObservation, MCioAction]):
    # The maximum change measured in pixels
    max_cursor_delta = 180.0 / env_util.DegreesToPixels.DEGREES_PER_PIXEL  # 1200

    def __init__(self, run_options: RunOptions, render_mode: str | None = None) -> None:
        """ """
        super().__init__(run_options=run_options, render_mode=render_mode)

        self.observation_space = spaces.Dict(
            {
                "frame": spaces.Box(
                    low=0,
                    high=255,
                    # shape=(height, width, channels)
                    shape=(self.run_options.height, self.run_options.width, 3),
                    dtype=np.uint8,
                ),
                "pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "pitch": spaces.Box(low=-90.0, high=90.0, shape=(1,), dtype=np.float32),
                "yaw": spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
            }
        )

        _action_space: dict[str, Any] = {
            key: spaces.Discrete(2) for key in INPUT_MAP.keys()
        }
        # Mouse movement in pixels relative to the current position
        _action_space["cursor_delta"] = spaces.Box(
            low=-self.max_cursor_delta,
            high=self.max_cursor_delta,
            shape=(2,),
            dtype=np.int32,
        )
        self.action_space = spaces.Dict(_action_space)

        # Env helpers
        self.input_mgr = env_util.InputStateManager()

    def _process_step(
        self, action: MCioAction, observation: MCioObservation
    ) -> tuple[int, bool, bool]:
        # reward, terminated, truncated
        return 0, False, False

    def _packet_to_observation(
        self, packet: mcio.network.ObservationPacket
    ) -> MCioObservation:
        """Convert an ObservationPacket to the environment observation_space"""
        obs: MCioObservation = {
            "frame": self.last_frame,
            "pos": env_util.nf32(packet.player_pos),
            "pitch": env_util.nf32(packet.player_pitch),
            "yaw": env_util.nf32(packet.player_yaw),
        }

        return obs

    def _action_to_packet(
        self, action: MCioAction, commands: list[str] | None = None
    ) -> mcio.network.ActionPacket:
        """Convert from the environment action_space to an ActionPacket"""
        packet = mcio.network.ActionPacket()
        packet.inputs = self.input_mgr.process_action(action, INPUT_MAP)
        if "cursor_delta" in action:
            rel_arr = action["cursor_delta"]
            dx, dy = rel_arr
            cursor_pos = (
                int(self.last_cursor_pos[0] + int(dx)),
                int(self.last_cursor_pos[1] + int(dy)),
            )
            packet.cursor_pos = [cursor_pos]

        packet.commands = commands or []

        return packet

    # Is there a better way to do this?
    def get_noop_action(self) -> MCioAction:
        action: MCioAction = {}
        for name in INPUT_MAP.keys():
            action[name] = NO_PRESS
        action["cursor_delta"] = CURSOR_DELTA_ZERO.copy()
        return action
