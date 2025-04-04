"""Defines some common types for the module"""

import enum
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Self

import glfw  # type: ignore

from . import config

# Project defines
DEFAULT_MINECRAFT_USER: Final[str] = "MCio"
DEFAULT_WINDOW_WIDTH: Final[int] = 854
DEFAULT_WINDOW_HEIGHT: Final[int] = 480
DEFAULT_ACTION_PORT: Final[int] = 4001  # 4ction
DEFAULT_OBSERVATION_PORT: Final[int] = 8001  # 8bservation
DEFAULT_HOST = "localhost"  # For security, only localhost
DEFAULT_HIDE_WINDOW = False


class StrEnumUpper(enum.StrEnum):
    """Like StrEnum, but the values are same as the enum rather than lowercase."""

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        return name


##
# Protocol types


class FrameType(StrEnumUpper):
    """Observation frame type. Currently just RAW."""

    RAW = enum.auto()


@dataclass
class InventorySlot:
    """Minecraft inventory slot - slot number, item id, and count"""

    slot: int
    id: str
    count: int


##
# Define types for key/button actions
# XXX Five types for storing 3 ints might be a bit much


class InputType(enum.IntEnum):
    KEY = 0
    MOUSE = 1


# GLFW key/button code, e.g. glfw.KEY_LEFT_SHIFT or glfw.MOUSE_BUTTON_LEFT
type GlfwCode = int


class GlfwAction(enum.IntEnum):
    RELEASE = glfw.RELEASE
    PRESS = glfw.PRESS
    # Note, not using glfw.REPEAT


@dataclass(frozen=True)  # Hashable
class InputID:
    type: InputType
    code: GlfwCode

    @classmethod
    def from_ints(cls, type_int: int, code: int) -> "InputID":
        return cls(type=InputType(type_int), code=code)


@dataclass(order=True)
class InputEvent:
    """Full input event sent to Minecraft"""

    type: InputType  # key / mouse
    code: GlfwCode  # glfw code
    action: GlfwAction  # press / release

    @classmethod
    def from_ints(cls, type_int: int, code: int, action_int: int) -> "InputEvent":
        """Alternate constructor that converts from int types to the enums."""
        return cls(type=InputType(type_int), code=code, action=GlfwAction(action_int))

    @classmethod
    def from_id(cls, input_id: InputID, action: GlfwAction) -> "InputEvent":
        return cls(type=input_id.type, code=input_id.code, action=action)


##


class MCioMode(StrEnumUpper):
    """MCio Mode"""

    OFF = enum.auto()
    ASYNC = enum.auto()
    SYNC = enum.auto()


DEFAULT_MCIO_MODE: Final[MCioMode] = MCioMode.ASYNC


@dataclass(kw_only=True)
class RunOptions:
    """Options for running Minecraft

    Args:
        instance_name: Set to launch an instance. None if connecting to a running instance.
        world_name: Launch directly into a world. You probably want to set this if instance_name is set.
            Otherwise you'll launch to the main menu.
        width: Frame width
        height: Frame height
        mcio_mode: sync/async
        hide_window: Don't show the Minecraft app window
        cleanup_on_signal: Kill the launched Minecraft and exit on SIGINT or SIGTERM
        action_port: port for action connection
        observation_port: port for observation connection
        mcio_dir: Top-level data directory
        java_path: Path to alternative java executable
        mc_username: Minecraft username
    """

    instance_name: config.InstanceName | None = None
    world_name: config.WorldName | None = None

    width: int = DEFAULT_WINDOW_WIDTH
    height: int = DEFAULT_WINDOW_HEIGHT
    mcio_mode: MCioMode = DEFAULT_MCIO_MODE
    hide_window: bool = DEFAULT_HIDE_WINDOW
    cleanup_on_signal: bool = True

    action_port: int = DEFAULT_ACTION_PORT
    observation_port: int = DEFAULT_OBSERVATION_PORT

    mcio_dir: Path | str = config.DEFAULT_MCIO_DIR
    java_path: str | None = None  # To use a different java executable

    mc_username: str = DEFAULT_MINECRAFT_USER
    instance_dir: Path | None = field(init=False)  # Auto-generated
    mc_uuid: uuid.UUID = field(init=False)  # Auto-generated

    def __post_init__(self) -> None:
        from . import instance

        self.mc_uuid: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_URL, self.mc_username)
        self.mcio_dir = Path(self.mcio_dir).expanduser()
        im = instance.InstanceManager(self.mcio_dir)
        if self.instance_name is not None:
            self.instance_dir = im.get_instance_dir(self.instance_name)
        else:
            self.instance_dir = None

    ##
    # Some simplified constructors for common cases

    @classmethod
    def for_launch(
        cls,
        instance_name: config.InstanceName,
        world_name: config.WorldName,
        width: int = DEFAULT_WINDOW_WIDTH,
        height: int = DEFAULT_WINDOW_HEIGHT,
    ) -> Self:
        """Simplified constructor for launching a Minecraft instance for an environment.
        This just creates the properly configured RunOptions. Launch using an env (or instance.Launcher).
        """

        return cls(
            instance_name=instance_name,
            world_name=world_name,
            width=width,
            height=height,
            mcio_mode=MCioMode.SYNC,
            hide_window=True,
        )

    @classmethod
    def for_connect(
        cls,
        width: int = DEFAULT_WINDOW_WIDTH,
        height: int = DEFAULT_WINDOW_HEIGHT,
    ) -> Self:
        """Simplified constructor for connecting to an already running Minecraft instance
        This just creates the properly configured RunOptions. Connect using an env
        """

        return cls(
            width=width,
            height=height,
            mcio_mode=MCioMode.SYNC,
            hide_window=True,
        )
