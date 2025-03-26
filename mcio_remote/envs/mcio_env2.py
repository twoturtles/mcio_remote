"""
This is a sample gym environment for MCio
"""

from dataclasses import dataclass
from typing import Any, Sequence

import glfw  # type: ignore
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

import mcio_remote as mcio
from mcio_remote.types import InputID, InputType

from . import env_util
from .base_env import McioBaseEnv, McioBaseEnvArgs

# Stub in the action and observation space types
type McioAction = dict[str, Any]
type McioObservation = dict[str, Any]


# Map from action name to Minecraft input. Mostly the same as minerl.
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
class McioEnvArgs(McioBaseEnvArgs):
    """See McioBaseEnvArgs for more info"""

    pass


class McioEnv(McioBaseEnv[McioObservation, McioAction]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    # The maximum change measured in pixels
    max_cursor_delta = 180.0 / env_util.DegreesToPixels.DEGREES_PER_PIXEL  # 1200

    def __init__(self, args: McioEnvArgs) -> None:
        """ """
        super().__init__(args)

        self.observation_space = spaces.Dict(
            {
                "frame": spaces.Box(
                    low=0,
                    high=255,
                    # shape=(height, width, channels)
                    shape=(self.run_options.height, self.run_options.width, 3),
                    dtype=np.uint8,
                ),
                "player_pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "pitch": spaces.Box(low=-90.0, high=90.0, shape=(), dtype=np.float32),
                "yaw": spaces.Box(low=-180.0, high=180.0, shape=(), dtype=np.float32),
            }
        )

        _action_space: dict[str, Any] = {
            key: spaces.Discrete(2) for key in INPUT_MAP.keys()
        }
        # Mouse movement in pixels relative to the current position
        _action_space["cursor_delta"] = (
            spaces.Box(
                low=-self.max_cursor_delta,
                high=self.max_cursor_delta,
                shape=(2,),
                dtype=np.int32,
            ),
        )
        self.action_space = spaces.Dict(_action_space)

        # Env helpers
        self.input_mgr = env_util.InputStateManager()

    def _process_step(
        self, action: McioAction, observation: McioObservation
    ) -> tuple[int, bool, bool]:
        # reward, terminated, truncated
        return 0, False, False

    def _packet_to_observation(
        self, packet: mcio.network.ObservationPacket
    ) -> McioObservation:
        """Convert an ObservationPacket to the environment observation_space"""
        self.last_frame = packet.get_frame_with_cursor()
        self.last_cursor_pos = packet.cursor_pos
        observation = {
            "frame": self.last_frame,
            "player_pos": _nf32(packet.player_pos),
            "player_pitch": _nf32(packet.player_pitch),
            "player_yaw": _nf32(packet.player_yaw),
        }
        return observation

    def _action_to_packet(
        self, action: McioAction, commands: list[str] | None = None
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


##
# Helper functions


def _nf32(seq: Sequence[int | float] | int | float) -> NDArray[np.float32]:
    """Convert to np.float32 arrays. Turns single values into 1D arrays."""
    if isinstance(seq, (int, float)):
        seq = [float(seq)]
    arr = np.array([float(val) for val in seq], dtype=np.float32)
    return arr
