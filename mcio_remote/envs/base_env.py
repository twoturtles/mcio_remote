"""Base class for MCio environments"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypedDict, TypeVar

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from mcio_remote import controller, gui, instance, network
from mcio_remote.types import RunOptions

# Reusable types
RenderFrame = TypeVar("RenderFrame")  # NDArray[np] shape = (height, width, channels)
ObsType = TypeVar("ObsType", bound=dict[str, Any])
ActType = TypeVar("ActType", bound=dict[str, Any])


class ResetOptions(TypedDict, total=False):
    """For now just commands"""

    commands: list[str]  # List of Minecraft commands


@dataclass
class McioBaseEnvArgs:
    """
    Wrap base class args in a class so child classes don't have to repeat them
    Args:
        run_options: Configuration options for MCio
        launch: Whether to launch a new Minecraft instance
        render_mode: The rendering mode (human, rgb_array)
    """

    run_options: RunOptions
    launch: bool = False
    render_mode: str | None = None


class McioBaseEnv(gym.Env[ObsType, ActType], Generic[ObsType, ActType], ABC):
    """Base class for MCio environments"""

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        args: McioBaseEnvArgs,
    ):
        """Base constructor for MCio environments

        Notes for subclasses:
         - Make sure you call super().__init__().
         - Set self.action_space and self.observation_space in the constructor.
         - Define _packet_to_observation(), _action_to_packet() and _process_step()
         - Optionally define _get_info()
        """
        self.run_options = args.run_options
        self.launch = args.launch
        assert (
            args.render_mode is None
            or args.render_mode in self.metadata["render_modes"]
        )
        self.render_mode = args.render_mode

        # Common state tracking
        self.last_frame: NDArray[np.uint8] | None = None

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
        """Implemented in subclasses. Convert an ObservationPacket to the environment observation_space"""
        pass

    @abstractmethod
    def _action_to_packet(
        self, action: ActType, commands: list[str] | None = None
    ) -> network.ActionPacket:
        """Implemented in subclasses. Convert from the environment action_space to an ActionPacket"""
        pass

    @abstractmethod
    def _process_step(
        self, action: ActType, observation: ObsType
    ) -> tuple[int, bool, bool]:
        """Implemented in subclasses. Called during step() after the observation has been received
        Returns (reward, terminated, truncated)"""
        pass

    def _get_info(self) -> dict[Any, Any]:
        """Optionally override this in subclasses. Used to return extra info from reset() and step()"""
        return {}

    ##
    # Internal methods

    def _get_obs(self) -> ObsType:
        """Receive an observation and pass it to the subclass.
        Sets self.last_frame to the most received frame."""
        assert self.ctrl is not None
        packet = self.ctrl.recv_observation()
        if packet is None:
            return {}
        self.last_frame = packet.get_frame_with_cursor()
        return self._packet_to_observation(packet)

    def _send_action(self, action: ActType, commands: list[str] | None = None) -> None:
        packet = self._action_to_packet(action, commands)
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
            if self.launcher is not None:
                # For multiple resets, close the previous connections, etc.
                self.close()
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
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}")
        options = options or ResetOptions()

        self._send_action(action, options.get("commands"))

        observation = self._get_obs()
        reward, terminated, truncated = self._process_step(action, observation)
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
