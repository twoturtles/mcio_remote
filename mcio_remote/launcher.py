"""Interface for managing and launching Minecraft instances"""

import argparse
import subprocess
import uuid
from pathlib import Path
import os
import pprint
import typing
from typing import Any, Final, Literal

from tqdm import tqdm
import requests

import minecraft_launcher_lib as mll

DEFAULT_MINECRAFT_DIR: Final[str] = "~/.mcio/minecraft"
DEFAULT_MINECRAFT_VERSION: Final[str] = "1.21.3"
DEFAULT_MINECRAFT_USER: Final[str] = "MCio"
DEFAULT_WINDOW_WIDTH: Final[int] = 854
DEFAULT_WINDOW_HEIGHT: Final[int] = 480

MCIO_MODE = Literal["off", "async", "sync"]

REQUIRED_MODS: list[str] = ["fabric-api", "mcio"]

# TODO
# multiple instances
# create world: name, seed, mode, difficulty, ...


class Launcher:
    """Launch Minecraft"""

    def __init__(
        self,
        mc_dir: Path | str | None = None,
        mc_username: str = DEFAULT_MINECRAFT_USER,
        mc_version: str = DEFAULT_MINECRAFT_VERSION,
        world: str | None = None,
        width: int = DEFAULT_WINDOW_WIDTH,
        height: int = DEFAULT_WINDOW_HEIGHT,
        mcio_mode: MCIO_MODE = "async",
    ) -> None:
        mc_dir = mc_dir or DEFAULT_MINECRAFT_DIR
        self.mc_dir = Path(mc_dir).expanduser()
        self.mc_username = mc_username
        self.mc_version = mc_version
        self.mc_uuid = uuid.uuid5(uuid.NAMESPACE_URL, self.mc_username)
        self.mcio_mode = mcio_mode

        # Store options
        options = mll.types.MinecraftOptions(
            username=mc_username,
            uuid=str(self.mc_uuid),
            token="MCioDev",
            customResolution=True,
            resolutionWidth=str(width),
            resolutionHeight=str(height),
        )
        if world:
            options["quickPlaySingleplayer"] = world
        self.mll_options = options

    def launch(self) -> None:
        env = self._get_env()
        cmd = self.get_command()
        # For some reason Minecraft logs end up in cwd, so set it to mc_dir
        subprocess.run(cmd, env=env, cwd=self.mc_dir)

    def get_command(self) -> list[str]:
        mc_cmd = mll.command.get_minecraft_command(
            self.mc_version, self.mc_dir, self.mll_options
        )
        mc_cmd = self._update_option_argument(mc_cmd, "--userType", "legacy")
        return mc_cmd

    def get_show_command(self) -> list[str]:
        """For testing, return the command that will be run"""
        cmd = [f"MCIO_MODE={self.mcio_mode}"]
        cmd += self.get_command()
        return cmd

    def _get_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["MCIO_MODE"] = self.mcio_mode
        return env

    def _update_option_argument(
        self, command_list: list[str], option: str, new_argument: str
    ) -> list[str]:
        try:
            new_list = command_list.copy()
            option_index = new_list.index(option)
            new_list[option_index + 1] = new_argument
            return new_list
        except ValueError:
            print(f"Option {option} not found in command list")
            raise
        except IndexError:
            print(f"Unexpected end of list after option {option}")
            raise


class Installer:
    """Install Minecraft along with Fabric and MCio"""

    def __init__(
        self,
        mc_dir: Path | str | None = None,
        mc_version: str = DEFAULT_MINECRAFT_VERSION,
    ) -> None:
        mc_dir = mc_dir or DEFAULT_MINECRAFT_DIR
        self.mc_dir = Path(mc_dir).expanduser()
        self.mc_version = mc_version

    def install(self) -> None:
        print("Installing Minecraft...")
        progress = _InstallProgress()
        mll.install.install_minecraft_version(
            self.mc_version, self.mc_dir, callback=progress.get_callbacks()
        )
        progress.close()

        print("\nInstalling Fabric...")
        progress = _InstallProgress()
        mll.fabric.install_fabric(
            self.mc_version, self.mc_dir, callback=progress.get_callbacks()
        )
        progress.close()

        # Install mods
        print()
        for mod in REQUIRED_MODS:
            self._install_mod(mod, self.mc_dir, self.mc_version)

        # XXX https://codeberg.org/JakobDev/minecraft-launcher-lib/issues/143
        err_path = self.mc_dir / "libraries/org/ow2/asm/asm/9.3/asm-9.3.jar"
        err_path.unlink()

        # Disable narrator
        opts = OptionsTxt(self.mc_dir / "options.txt")
        opts["narrator"] = "0"
        opts.save()

    def _install_mod(
        self, mod_id: str, mc_dir: Path, mc_ver: str, version_type: str = "release"
    ) -> None:
        mod_info_url = f'https://api.modrinth.com/v2/project/{mod_id}/version?game_versions=["{mc_ver}"]'
        response = requests.get(mod_info_url)
        response.raise_for_status()
        info_list: list[Any] = response.json()

        found: dict[str, Any] | None = None
        for vers_info in info_list:
            if vers_info["version_type"] == version_type:
                found = vers_info
                break

        if not found:
            raise ValueError(
                f"No {version_type} version found for {mod_id} supporting Minecraft {mc_ver}"
            )
        # Is the jar always the first in the "files" list?
        jar_info = found["files"][0]
        response = requests.get(jar_info["url"])
        response.raise_for_status()
        filename = jar_info["filename"]

        mods_dir = mc_dir / "mods"
        mods_dir.mkdir(exist_ok=True)
        print(f"Installing {filename}")
        with open(mods_dir / filename, "wb") as f:
            f.write(response.content)


class _InstallProgress:
    """Progress bar for Minecraft installer"""

    def __init__(self, desc_width: int = 40) -> None:
        self.pbar: tqdm[Any] | None = None
        self.desc_width = desc_width
        self.current = 0

    def get_callbacks(self) -> mll.types.CallbackDict:
        return mll.types.CallbackDict(
            setStatus=self._set_status,
            setProgress=self._set_progress,
            setMax=self._set_max,
        )

    def close(self) -> None:
        if self.pbar:
            self.pbar.close()

    def _set_max(self, total: int) -> None:
        """The installer calls set_max multiple times. Create a new bar each time."""
        if self.pbar:
            self.pbar.close()
        self.pbar = tqdm(total=total)
        self.current = 0

    def _set_status(self, status: str) -> None:
        if self.pbar:
            status = status[: self.desc_width].ljust(self.desc_width)
            self.pbar.set_description(status)

    def _set_progress(self, current: int) -> None:
        if self.pbar:
            self.pbar.update(current - self.current)
            self.current = current


class OptionsTxt:
    """Load/Save options.txt. Keeps everything as strings."""

    def __init__(self, options_path: Path | str) -> None:
        self.path = Path(options_path).expanduser()
        self.options = self._load(self.path)

    def save(self) -> None:
        """Save options back to file"""
        with self.path.open("w") as f:
            for key, value in self.options.items():
                f.write(f"{key}:{value}\n")

    def __getitem__(self, key: str) -> str:
        return self.options[key]

    def __setitem__(self, key: str, value: str) -> None:
        self.options[key] = value

    def _load(self, options_path: Path) -> dict[str, str]:
        """Load options from file"""
        if not self.path.exists():
            return {}

        with self.path.open("r") as f:
            txt = f.read()
        lines = txt.strip().split()
        options = {}
        for line in lines:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            options[key] = value
        return options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minecraft Instance Manager and Launcher"
    )

    # Subparsers for different modes
    subparsers = parser.add_subparsers(dest="cmd_mode", required=True)

    # Utility to add common arguments
    def add_arguments(
        parser: argparse.ArgumentParser, args: list[dict[str, Any]]
    ) -> None:
        for arg in args:
            parser.add_argument(*arg["flags"], **arg["kwargs"])

    # Common arguments
    minecraft_dir_arg = {
        "flags": ["--minecraft-dir", "-d"],
        "kwargs": {
            "type": str,
            "help": f"Minecraft directory (default: {DEFAULT_MINECRAFT_DIR})",
        },
    }

    minecraft_ver_arg = {
        "flags": ["--version", "-v"],
        "kwargs": {
            "type": str,
            "default": DEFAULT_MINECRAFT_VERSION,
            "help": f"Minecraft version to install/launch (default: {DEFAULT_MINECRAFT_VERSION})",
        },
    }

    username_arg = {
        "flags": ["--username", "-u"],
        "kwargs": {
            "type": str,
            "default": DEFAULT_MINECRAFT_USER,
            "help": f"Player name (default: {DEFAULT_MINECRAFT_USER})",
        },
    }

    world_arg = {
        "flags": ["--world", "-w"],
        "kwargs": {
            "type": str,
            "help": "World name",
        },
    }

    width_arg = {
        "flags": ["--width", "-W"],
        "kwargs": {
            "type": int,
            "default": DEFAULT_WINDOW_WIDTH,
            "help": f"Window width (default: {DEFAULT_WINDOW_WIDTH})",
        },
    }

    height_arg = {
        "flags": ["--height", "-H"],
        "kwargs": {
            "type": int,
            "default": DEFAULT_WINDOW_HEIGHT,
            "help": f"Window height (default: {DEFAULT_WINDOW_HEIGHT})",
        },
    }

    mcio_mode_arg = {
        "flags": ["--mcio_mode", "-m"],
        "kwargs": {
            "type": str,
            "choices": typing.get_args(MCIO_MODE),
            "default": "async",
            "help": "MCIO mode: (default: async)",
        },
    }

    ##
    # Install subparser
    install_parser = subparsers.add_parser("install", help="Install Minecraft")
    add_arguments(install_parser, [minecraft_dir_arg, minecraft_ver_arg])

    ##
    # Launch subparser
    launch_parser = subparsers.add_parser("launch", help="Launch Minecraft")
    add_arguments(
        launch_parser,
        [
            minecraft_dir_arg,
            minecraft_ver_arg,
            mcio_mode_arg,
            world_arg,
            width_arg,
            height_arg,
            username_arg,
        ],
    )

    launch_group = launch_parser.add_mutually_exclusive_group()
    launch_group.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="Don't run the command; print it as a list",
    )
    launch_group.add_argument(
        "--str",
        action="store_true",
        default=False,
        help="Don't run the command; print it as a string",
    )

    ##
    # Show subparser
    show_parser = subparsers.add_parser(
        "show", help="Show information about what is installed"
    )
    add_arguments(show_parser, [minecraft_dir_arg])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.cmd_mode == "install":
        installer = Installer(args.minecraft_dir, args.version)
        installer.install()
    elif args.cmd_mode == "launch":
        launcher = Launcher(
            mc_dir=args.minecraft_dir,
            mc_username=args.username,
            mc_version=args.version,
            world=args.world,
            width=args.width,
            height=args.height,
            mcio_mode=args.mcio_mode,
        )
        if args.list:
            cmd = launcher.get_show_command()
            pprint.pprint(cmd)
        elif args.str:
            cmd = launcher.get_show_command()
            print(" ".join(cmd))
        else:
            launcher.launch()
    elif args.cmd_mode == "show":
        # TODO
        for info in mll.utils.get_installed_versions(args.mc_dir):
            pprint.pprint(info)
    else:
        print(f"Unknown mode: {args.cmd_mode}")
