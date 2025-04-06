"""Base class for MCio environments"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypedDict, TypeVar

import glfw  # type: ignore
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from mcio_remote import controller, gui, instance, network, types

# Reusable types
RenderFrame = TypeVar("RenderFrame")  # NDArray[np] shape = (height, width, channels)
ObsType = TypeVar("ObsType", bound=dict[str, Any])
ActType = TypeVar("ActType", bound=dict[str, Any])


class ResetOptions(TypedDict, total=False):
    """For now just commands
    valid options:
    commands: list[str]
        List of server commands to initialize the environment.
        E.g. teleport, time set, etc. Do not include the initial "/" in the commands.

        Note: Different command types seem to take different amounts of time to
        execute in Minecraft. You may want to use skip_ticks() after commands to
        make sure they have taken effect. I've seen ~20 ticks before a "time
        set" command takes effect.
    """

    commands: list[str]


class MCioBaseEnv(gym.Env[ObsType, ActType], Generic[ObsType, ActType], ABC):
    """Base class for MCio environments
    Notes for subclasses:
        - Make sure you call super().__init__().
        - Set self.action_space and self.observation_space in the constructor.
        - Define _packet_to_observation(), _action_to_packet() and _process_step()
        - Optionally define _get_info()

    Along with the callbacks, self.last_frame and self.last_cursor_pos are available for subclasses.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self, run_options: types.RunOptions, render_mode: str | None = None
    ) -> None:
        """Base constructor for MCio environments
        Args:
            run_options: Configuration options for MCio. If instance_name is set, a Minecraft
                instance will be started, otherwise the environment will connect to a previously launched instance.
                See mcio_remote.types.RunOptions
            render_mode: The rendering mode (human, rgb_array)

        """
        self.run_options = run_options
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Common state tracking
        self.last_frame: NDArray[np.uint8] | None = None
        self.last_cursor_pos: tuple[int, int] = (0, 0)

        # These need closing when done. Handled in close().
        self.gui: gui.ImageStreamGui | None = None
        self.ctrl: controller.ControllerCommon | None = None
        self.launcher: instance.Launcher | None = None

        # Define spaces in subclasses
        self.action_space: gym.spaces.Space[ActType]
        self.observation_space: gym.spaces.Space[ObsType]

    ##
    # Define in subclasses

    @abstractmethod
    def _packet_to_observation(self, packet: network.ObservationPacket) -> ObsType:
        """Convert an ObservationPacket to the environment observation_space"""
        pass

    @abstractmethod
    def _action_to_packet(
        self, action: ActType, commands: list[str] | None = None
    ) -> network.ActionPacket:
        """Convert from the environment action_space to an ActionPacket"""
        pass

    @abstractmethod
    def _process_step(
        self, action: ActType, observation: ObsType
    ) -> tuple[int, bool, bool]:
        """Called during step() after the observation has been received.
        Returns (reward, terminated, truncated)"""
        pass

    def _get_info(self) -> dict[Any, Any]:
        """Optionally override this in subclasses. Used to return extra info from reset() and step()"""
        return {}

    ##
    # Internal methods

    def _get_obs(self) -> ObsType:
        """Receive an observation and pass it to the subclass.
        Updates self.last_frame self.last_cursor_pos"""
        assert self.ctrl is not None
        packet = self.ctrl.recv_observation()

        self.last_frame = packet.get_frame_with_cursor()
        self.last_cursor_pos = packet.cursor_pos

        # Call to subclass
        obs = self._packet_to_observation(packet)
        # assert obs in self.observation_space
        return obs

    def _send_action(self, action: ActType, commands: list[str] | None = None) -> None:
        # Call to subclass
        packet = self._action_to_packet(action, commands)
        # assert action in self.action_space
        assert self.ctrl is not None
        self.ctrl.send_action(packet)

    def _send_reset_action(self, options: ResetOptions) -> None:
        """Clear inputs and send initialization commands"""
        packet = network.ActionPacket(
            clear_input=True, commands=options.get("commands", [])
        )
        assert self.ctrl is not None
        self.ctrl.send_action(packet)

    def reset(
        self,
        seed: int | None = None,
        *,
        options: ResetOptions | None = None,  # type: ignore[override]
    ) -> tuple[ObsType, dict[Any, Any]]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        options = options or ResetOptions()

        # For multiple resets, close any previous connections, etc.
        self.close()

        if self.run_options.instance_name is not None:
            self.launcher = instance.Launcher(self.run_options)
            self.launcher.launch(wait=False)

        if self.run_options.mcio_mode == "async":
            self.ctrl = controller.ControllerAsync()
        else:
            self.ctrl = controller.ControllerSync()

        # The reset action will trigger an initial observation
        self._send_reset_action(options)
        observation = self._get_obs()
        info = self._get_info()

        # XXX This is from the official gymnasium template, but why
        # is this here? Shouldn't the user just call render after reset?
        if self.render_mode == "human":
            self._render_frame_human()

        return observation, info

    def step(
        self,
        action: ActType,
        *,
        options: ResetOptions | None = None,
    ) -> tuple[ObsType, int, bool, bool, dict[Any, Any]]:
        """Env step function. Includes extra options arg to allow command to be sent during step."""
        options = options or ResetOptions()

        self._send_action(action, options.get("commands"))

        observation = self._get_obs()

        reward, terminated, truncated = self._process_step(action, observation)
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame_human()

        return observation, reward, terminated, truncated, info

    def skip_ticks(
        self, n_steps: int
    ) -> tuple[ObsType, int, bool, bool, dict[Any, Any]]:
        """Send empty actions and return the final observation. Use to skip over
        a number of steps/game ticks"""
        assert self.ctrl is not None
        pkt = network.ActionPacket()
        for i in range(n_steps):
            self.ctrl.send_action(pkt)
            observation = self._get_obs()
        # observation, reward, terminated, truncated, info
        return observation, 0, False, False, {}

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
        """This supports multiple closes / resets"""
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
    # Debug helpers

    def step_raw(self, pkt: network.ActionPacket) -> network.ObservationPacket:
        """Expose sending raw actions"""
        assert self.ctrl is not None
        self.ctrl.send_action(pkt)
        return self.ctrl.recv_observation()

    def toggle_f3(self) -> None:
        """Toggle the debug screen"""
        pkt = network.ActionPacket()
        f3 = types.InputID(types.InputType.KEY, glfw.KEY_F3)
        pkt.inputs = [
            types.InputEvent.from_id(f3, types.GlfwAction.PRESS),
            types.InputEvent.from_id(f3, types.GlfwAction.RELEASE),
        ]
        self.step_raw(pkt)
