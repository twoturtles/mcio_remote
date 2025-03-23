"""This is a sample gym environment for MCio. The current plan is to make an environment
base class for MCio once the requirements become more clear"""

from typing import Any, Sequence, TypedDict, TypeVar

import glfw  # type: ignore
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from mcio_remote import controller, gui, instance, network, types

##
# Defines used in creating spaces

# Define the subset of all keys/buttons that we're using
# Automate the mapping?
MINECRAFT_KEYS = {
    "W": glfw.KEY_W,
    "A": glfw.KEY_A,
    "S": glfw.KEY_S,
    "D": glfw.KEY_D,
    "E": glfw.KEY_E,
    "SPACE": glfw.KEY_SPACE,
    "L_SHIFT": glfw.KEY_LEFT_SHIFT,
}

MINECRAFT_MOUSE_BUTTONS = {
    "LEFT": glfw.MOUSE_BUTTON_LEFT,
    "RIGHT": glfw.MOUSE_BUTTON_RIGHT,
}

# key / button states in action spaces
NO_PRESS = np.int64(0)
PRESS = np.int64(1)

CURSOR_REL_BOUND_DEFAULT = 1000
NO_CURSOR_REL = np.array((0.0, 0.0), dtype=np.float32)
NO_CURSOR_REL.flags.writeable = False

# XXX gymnasium.utils.env_checker.check_env

# Stub in the action and observation space types
type MCioAction = dict[str, Any]
type MCioObservation = dict[str, Any]
RenderFrame = TypeVar("RenderFrame")  # NDArray[np] shape = (height, width, channels)


class ResetOptions(TypedDict, total=False):
    """For now just commands"""

    commands: list[str]  # List of Minecraft commands


class MCioEnv(gym.Env[MCioObservation, MCioAction]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }  # XXX Copied from gym sample env.

    def __init__(
        self,
        run_options: types.RunOptions,
        *,
        launch: bool = False,
        cursor_rel_bound: int = CURSOR_REL_BOUND_DEFAULT,
        render_mode: str | None = None,
    ):
        """Model gym environment

        Args:
            run_options:
                If you're not using this env to launch Minecraft, the only options
                used are height, width, and mcio_mode.

                The remaining options are used if the env is launching an instance. At least
                instance_name is required in that case.

            launch: Should the env launch Minecraft

            cursor_rel_bound:

                This defines the size of the cursor_pos_rel Box space. Essentially,
                the maximum distance the cursor can move in a particular direction in one step.

            render_mode: human, rgb_array
        """
        self.run_options = run_options
        self.launch = launch
        self.cursor_rel_bound = cursor_rel_bound
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.last_frame: NDArray[np.uint8] | None = None
        self.last_cursor_pos: tuple[int, int] = (0, 0)
        self.keys_pressed: set[str] = set()
        self.mouse_buttons_pressed: set[str] = set()

        # These need closing when done. Handled in close().
        self.gui: gui.ImageStreamGui | None = None
        self.ctrl: controller.ControllerCommon | None = None
        self.launcher: instance.Launcher | None = None

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
                    low=_nf32([-np.inf, -np.inf, -np.inf]),
                    high=_nf32([np.inf, np.inf, np.inf]),
                ),
                "player_pitch": spaces.Box(low=_nf32(-90), high=_nf32(90)),
                "player_yaw": spaces.Box(low=_nf32(-180), high=_nf32(180)),
            }
        )

        self.action_space = spaces.Dict(
            {
                # For keys and mouse buttons, 1 = pressed, 0 = not pressed
                "keys": spaces.Dict(
                    {key: spaces.Discrete(2) for key in MINECRAFT_KEYS.keys()}
                ),
                "mouse_buttons": spaces.Dict(
                    {
                        button: spaces.Discrete(2)
                        for button in MINECRAFT_MOUSE_BUTTONS.keys()
                    }
                ),
                # Mouse movement relative to the current position
                # Change to minerl camera setup?
                "cursor_pos_rel": spaces.Box(
                    low=-cursor_rel_bound,
                    high=cursor_rel_bound,
                    shape=(2,),
                ),
            }
        )

    # Is there a better way to get a noop? Wrappers?
    # E.g., noop = env.unwrapped.get_noop_action() XXX Don't require unwrapped
    def get_noop_action(self) -> dict[str, Any]:
        action: MCioAction = {}

        action["keys"] = {}
        for name in MINECRAFT_KEYS.keys():
            action["keys"][name] = NO_PRESS

        action["mouse_buttons"] = {}
        for name in MINECRAFT_MOUSE_BUTTONS.keys():
            action["mouse_buttons"][name] = NO_PRESS

        action["cursor_pos_rel"] = NO_CURSOR_REL.copy()

        assert action in self.action_space
        return action

    def _get_obs(self) -> MCioObservation:
        assert self.ctrl is not None
        packet = self.ctrl.recv_observation()
        if packet is None:
            return {}
        return self._packet_to_observation(packet)

    def _send_action(
        self, action: MCioAction, commands: list[str] | None = None
    ) -> None:
        packet = self._action_to_packet(action, commands)
        assert self.ctrl is not None
        self.ctrl.send_action(packet)

    def _packet_to_observation(
        self, packet: network.ObservationPacket
    ) -> MCioObservation:
        """Convert an ObservationPacket to the environment observation_space
        XXX Sets self.last_frame and self.last_cursor_pos as side-effects"""
        # Convert all fields to numpy arrays with correct dtypes
        self.last_frame = packet.get_frame_with_cursor()
        self.last_cursor_pos = packet.cursor_pos
        observation = {
            "frame": self.last_frame,
            "player_pos": _nf32(packet.player_pos),
            "player_pitch": _nf32(packet.player_pitch),
            "player_yaw": _nf32(packet.player_yaw),
        }
        return observation

    # XXX I think missing keys/buttons should translate to NO_PRESS. But what is noop then?
    # Convert action space values to MCio/Minecraft values. Allow for empty/noop actions.
    def _action_to_packet(
        self, action: MCioAction, commands: list[str] | None = None
    ) -> network.ActionPacket:
        """Convert from the environment action_space to an ActionPacket"""
        packet = network.ActionPacket()
        if action is None and commands is None:  # noop
            return packet

        action = action or {}
        commands = commands or []
        packet.commands = commands

        # Convert action_space key indices to Minecraft (key, action) pairs
        input_list: list[types.InputEvent] = []
        if "keys" in action:
            keys_list = self._space_map_to_packet(
                action["keys"], MINECRAFT_KEYS, self.keys_pressed
            )
            input_list += [
                types.InputEvent.from_ints(types.InputType.KEY, x[0], x[1])
                for x in keys_list
            ]

        # Convert action_space mouse button indices to Minecraft (button, action) pairs
        if "mouse_buttons" in action:
            buttons_list = self._space_map_to_packet(
                action["mouse_buttons"],
                MINECRAFT_MOUSE_BUTTONS,
                self.mouse_buttons_pressed,
            )
            input_list += [
                types.InputEvent.from_ints(types.InputType.MOUSE, x[0], x[1])
                for x in buttons_list
            ]

        packet.inputs = input_list

        # Convert cursor position
        if "cursor_pos_rel" in action:
            rel_arr = action["cursor_pos_rel"]
            if not np.array_equal(rel_arr, NO_CURSOR_REL):
                dx, dy = rel_arr
                cursor_pos = (
                    int(self.last_cursor_pos[0] + int(dx)),
                    int(self.last_cursor_pos[1] + int(dy)),
                )
                packet.cursor_pos = [cursor_pos]

        return packet

    def _space_map_to_packet(
        self,
        space_dict: dict[str, np.int64],
        conv_dict: dict[str, int],
        pressed_set: set[str],
    ) -> list[tuple[int, int]]:
        """Map keys and buttons in the action space to Minecraft press/release
        Also updates self.keys_pressed and self.mouse_buttons_pressed
        Returns list of [glfw_code, glfw_action]"""
        pairs = []
        for name, action in space_dict.items():
            pressed = bool(action)  # 1 = pressed, 0 = not pressed
            assert name in conv_dict  # in MINECRAFT_KEYS or MINECRAFT_MOUSE_BUTTONS
            glfw_code = conv_dict[name]
            if pressed == (name in pressed_set):
                # No change. Should continued pressing generate REPEAT?
                continue
            if pressed:
                pair = (glfw_code, glfw.PRESS)
                pressed_set.add(name)
            else:
                pair = (glfw_code, glfw.RELEASE)
                pressed_set.remove(name)
            pairs.append(pair)

        return pairs

    def _get_reset_action(self) -> dict[str, Any]:
        """This creates an action that resets all input"""
        action: MCioAction = {}

        action["keys"] = {}
        for name in MINECRAFT_KEYS.keys():
            action["keys"][name] = NO_PRESS

        action["mouse_buttons"] = {}
        for name in MINECRAFT_MOUSE_BUTTONS.keys():
            action["mouse_buttons"][name] = NO_PRESS

        # Move cursor back to (0, 0). Make configurable - face East, etc.?
        action["cursor_pos_rel"] = np.array(
            (-self.last_cursor_pos[0], -self.last_cursor_pos[1]), dtype=np.float32
        )

        assert action in self.action_space
        return action

    def _get_info(self) -> dict[Any, Any]:
        return {}

    def reset(
        self,
        seed: int | None = None,
        *,
        options: ResetOptions | None = None,  # type: ignore[override]
    ) -> tuple[MCioObservation, dict[Any, Any]]:
        """valid options:
        commands: list[str] | None = None
            List of server commands to initialize the environment.
            E.g. teleport, time set, etc. Do not include the initial "/" in the commands.
        launcher_options: instance:LauncherOptions | None = None
            Minecraft launch options. Will skip launch if None.
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        options = options or ResetOptions()

        if self.launch:
            # For multiple resets, close the previous connections, etc.
            if self.launcher is not None:
                self.close()
            self.launcher = instance.Launcher(self.run_options)
            self.launcher.launch(wait=False)

        if self.run_options.mcio_mode == "async":
            self.ctrl = controller.ControllerAsync()
        else:
            self.ctrl = controller.ControllerSync()

        commands = options.get("commands")
        # Send reset action with initialization commands to trigger an observation
        self._send_action(self._get_reset_action(), commands)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame_human()

        return observation, info

    def step(
        self,
        action: MCioAction,
        *,
        options: ResetOptions | None = None,
    ) -> tuple[MCioObservation, int, bool, bool, dict[Any, Any]]:
        """Env step function. Includes extra options arg to allow command to be sent during step."""
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}")
        options = options or ResetOptions()

        self._send_action(action, options.get("commands"))

        observation = self._get_obs()
        reward = 0
        terminated = False
        truncated = False
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame_human()

        return observation, reward, terminated, truncated, info

    # NDArray[np.uint8] shape = (height, width, channels)
    # Gym's render returns a generic TypeVar("RenderFrame"), which is not very useful.
    def render(self) -> NDArray[np.uint8] | None:  # type: ignore[override]
        if self.render_mode == "human":
            self._render_frame_human()
        elif self.render_mode == "rgb_array":
            return self._render_frame_rgb_array()
        return None

    def _render_frame_rgb_array(self) -> NDArray[np.uint8] | None:
        return self.last_frame

    def _render_frame_human(self) -> None:
        if self.gui is None and self.render_mode == "human":
            self.gui = gui.ImageStreamGui(
                "MCio", width=self.run_options.width, height=self.run_options.height
            )
        if self.last_frame is None:
            return
        assert self.gui is not None
        self.gui.show(self.last_frame)

    def close(self) -> None:
        if self.gui is not None:
            self.gui.close()
            self.gui = None
        if self.ctrl is not None:
            if self.launcher is not None:
                # If we launched Minecraft, try for a clean exit.
                self.ctrl.send_stop()
            self.ctrl.close()
            self.ctrl = None
        if self.launcher is not None:
            self.launcher.close()
            self.launcher = None


##
# Helper functions


def _nf32(seq: Sequence[int | float] | int | float) -> NDArray[np.float32]:
    """Convert to np.float32 arrays. Turns single values into 1D arrays."""
    if isinstance(seq, (int, float)):
        seq = [float(seq)]
    arr = np.array([float(val) for val in seq], dtype=np.float32)
    return arr
