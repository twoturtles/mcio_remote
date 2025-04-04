"""Interface for managing and launching Minecraft instances"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Any, Final

import minecraft_launcher_lib as mll
import requests

from . import config, types, util

LOG = logging.getLogger(__name__)

INSTANCES_SUBDIR: Final[str] = "instances"
REQUIRED_MODS: Final[tuple[str, ...]] = ("fabric-api", "mcio")

# XXX Rethink classes - Installer / Launcher / InstanceManager are confusing


class Installer:
    """Install Minecraft along with Fabric and MCio"""

    def __init__(
        self,
        instance_name: "config.InstanceName",
        mcio_dir: Path | str | None = None,
        mc_version: str = config.DEFAULT_MINECRAFT_VERSION,
        java_path: str | None = None,  # Used to run the Fabric installer
    ) -> None:
        self.instance_name = instance_name
        mcio_dir = mcio_dir or config.DEFAULT_MCIO_DIR
        self.mcio_dir = Path(mcio_dir).expanduser()
        self.mc_version = mc_version
        im = InstanceManager(self.mcio_dir)
        self.instance_dir = im.get_instance_dir(self.instance_name)
        self.java_path = java_path

        with config.ConfigManager(self.mcio_dir) as cfg_mgr:
            if cfg_mgr.config.instances.get(self.instance_name) is not None:
                print(
                    f"Warning: Instance {self.instance_name} already exists in {cfg_mgr.config_file}"
                )

    def install(self) -> None:
        print(f"Installing Minecraft {self.mc_version} in {self.instance_dir}...")
        progress = util.InstallProgress()
        # mll install uses more threads than connections, so urllib3 gives a warning.
        # It's harmless, so silence the warning.
        logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
        mll.install.install_minecraft_version(
            self.mc_version, self.instance_dir, callback=progress.get_callbacks()
        )
        progress.close()

        print("\nInstalling Fabric...")
        # Use the Minecraft jvm to install Fabric.
        jvm_info = mll.runtime.get_version_runtime_information(
            self.mc_version, self.instance_dir
        )
        assert jvm_info is not None
        # jvm_info = {'name': 'java-runtime-delta', 'javaMajorVersion': 21}
        if self.java_path is None:
            java_cmd = mll.runtime.get_executable_path(
                jvm_info["name"], self.instance_dir
            )
        else:
            java_cmd = self.java_path

        progress = util.InstallProgress()
        # XXX This doesn't check that the loader is compatible with the minecraft version
        fabric_ver = mll.fabric.get_latest_loader_version()
        mll.fabric.install_fabric(
            self.mc_version,
            self.instance_dir,
            loader_version=fabric_ver,
            callback=progress.get_callbacks(),
            java=java_cmd,
        )
        progress.close()
        # This is the format mll uses to generate the version string.
        # XXX Would prefer to get this automatically.
        fabric_minecraft_version = f"fabric-loader-{fabric_ver}-{self.mc_version}"

        # Install mods
        print()
        for mod in REQUIRED_MODS:
            self._install_mod(mod, self.instance_dir, self.mc_version)

        # Disable narrator
        with util.OptionsTxt(self.instance_dir / "options.txt", save=True) as opts:
            opts["narrator"] = "0"

        with config.ConfigManager(self.mcio_dir, save=True) as cfg_mgr:
            cfg_mgr.config.instances[self.instance_name] = config.InstanceConfig(
                name=self.instance_name,
                launch_version=fabric_minecraft_version,
                minecraft_version=self.mc_version,
            )
        print("Success!")

    def _install_mod(
        self,
        mod_id: str,
        instance_dir: Path,
        mc_ver: str,
        version_type: str = "release",
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

        mods_dir = instance_dir / "mods"
        mods_dir.mkdir(parents=True, exist_ok=True)
        print(f"Installing {filename}")
        with open(mods_dir / filename, "wb") as f:
            f.write(response.content)


class Launcher:
    """Launch Minecraft"""

    def __init__(self, run_options: types.RunOptions) -> None:

        self.run_options = run_options
        if self.run_options.instance_name is None:
            raise ValueError("instance_name is required")

        with config.ConfigManager(self.run_options.mcio_dir) as cm:
            instance_config = cm.config.instances.get(self.run_options.instance_name)
        if instance_config is None:
            raise ValueError(
                f"Instancd {self.run_options.instance_name} not found in {cm.config_file}"
            )
        self.launch_version = instance_config.launch_version

        # Store options
        mll_opts = mll.types.MinecraftOptions(
            username=self.run_options.mc_username,
            uuid=str(self.run_options.mc_uuid),
            token="MCioDev",
            customResolution=True,
            resolutionWidth=str(self.run_options.width),
            resolutionHeight=str(self.run_options.height),
        )
        if self.run_options.world_name is not None:
            mll_opts["quickPlaySingleplayer"] = self.run_options.world_name
        if self.run_options.java_path is not None:
            mll_opts["executablePath"] = self.run_options.java_path

        # XXX Hack in a way to pass a log4j config file to Minecraft
        log_cfg = os.environ.get("MCIO_LOG_CFG")
        if log_cfg is not None:
            log_path = Path(log_cfg).resolve()
            mll_opts["jvmArguments"] = [f"-Dlog4j.configurationFile={log_path}"]

        self.mll_opts = mll_opts

        self._process: subprocess.Popen[str] | None = None
        self._in_wait: bool = False
        if self.run_options.cleanup_on_signal:
            # Note, python only allows signal handlers on the main thread. So if you're
            # running this on another thread, set cleanup_on_signal to False.
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    def launch(self, wait: bool = False) -> None:
        """launch the instance

        Args:
            wait: block waiting for the instance to exit
        """
        env = self._get_env()
        cmd = self.get_command()
        # For some reason Minecraft logs end up in cwd, so set it to instance_dir
        self._process = subprocess.Popen(
            cmd, env=env, cwd=self.run_options.instance_dir, text=True
        )
        if wait:
            LOG.info(f"Wait-on-pid {self._process.pid}")
            self._in_wait = True
            self._process.wait()
            self._in_wait = False
            self._process = None

    def _close_wait(self) -> None:
        LOG.info("Close-Wait")
        assert self._process is not None
        self._process.terminate()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

    def _close_no_wait(self) -> None:
        LOG.info("Close-No-Wait")
        assert self._process is not None
        self._process.terminate()
        # Give it a second to exit, then kill.
        # Could improve this with psutil to see if the process is a zombie.
        time.sleep(1)
        self._process.kill()

    def close(self) -> None:
        if self._process is None:
            return
        LOG.info(f"Closing-Subprocess pid={self._process.pid}")
        if self._in_wait:
            # You can't call wait twice due to an internal lock. If we're
            # already in wait in __init__(), don't try to wait here.
            self._close_no_wait()
        else:
            self._close_wait()
        LOG.info("Close-Complete")
        self._process = None

    def poll(self) -> int | None:
        """Return the process return code, or None if still running"""
        assert self._process is not None
        return self._process.poll()

    def wait(self) -> None:
        if self._process is not None:
            self._process.wait()
            self._process = None

    def get_command(self) -> list[str]:
        assert self.run_options.instance_dir is not None
        mc_cmd = mll.command.get_minecraft_command(
            self.launch_version, self.run_options.instance_dir, self.mll_opts
        )
        mc_cmd = self._update_option_argument(mc_cmd, "--userType", "legacy")
        return mc_cmd

    def get_show_command(self) -> list[str]:
        """For testing, return the command that will be run"""
        env = self._get_env_options()
        cmd = [f"{key}={value}" for key, value in env.items()]
        cmd += self.get_command()
        return cmd

    def _get_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self._get_env_options())
        return env

    def _get_env_options(self) -> dict[str, str]:
        env: dict[str, str] = {}
        env["MCIO_MODE"] = self.run_options.mcio_mode
        env["MCIO_ACTION_PORT"] = str(self.run_options.action_port)
        env["MCIO_OBSERVATION_PORT"] = str(self.run_options.observation_port)
        env["MCIO_HIDE_WINDOW"] = str(self.run_options.hide_window)
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

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:
        """I want to be able to ctrl-c the python process that launched Minecraft and have Minecraft exit"""
        signame = signal.Signals(signum).name
        LOG.info(f"Received-Signal {signame} ({signum})")
        self.close()
        sys.exit(0)


##
# Utility functions
class InstanceManager:
    def __init__(
        self,
        mcio_dir: Path | str | None = None,
    ) -> None:
        mcio_dir = mcio_dir or config.DEFAULT_MCIO_DIR
        self.mcio_dir = Path(mcio_dir).expanduser()

    def get_instances_dir(self) -> Path:
        return self.mcio_dir / INSTANCES_SUBDIR

    def get_instance_dir(self, instance_name: config.InstanceName) -> Path:
        return self.get_instances_dir() / instance_name

    def get_saves_dir(self, instance_name: config.InstanceName) -> Path:
        SAVES_SUBDIR = "saves"
        instance_dir = self.get_instance_dir(instance_name)
        return instance_dir / SAVES_SUBDIR

    def instance_exists(self, instance_name: config.InstanceName) -> bool:
        instance_dir = self.get_instance_dir(instance_name)
        return instance_dir.exists()

    def get_instance_world_list(self, instance_name: config.InstanceName) -> list[str]:
        world_dir = self.get_saves_dir(instance_name)
        world_names = [x.name for x in world_dir.iterdir() if x.is_dir()]
        return world_names

    def copy(
        self,
        src: config.InstanceName,
        dst: config.InstanceName,
        overwrite: bool = False,
    ) -> None:
        src_dir = self.get_instance_dir(src)
        dst_dir = self.get_instance_dir(dst)
        util.copy_dir(src_dir, dst_dir, overwrite=overwrite)
        with config.ConfigManager(self.mcio_dir, save=True) as cm:
            cm.config.instances[dst] = cm.config.instances[src]

    def delete(self, instance_name: config.InstanceName) -> None:
        instance_dir = self.get_instance_dir(instance_name)
        util.rmrf(instance_dir)
        with config.ConfigManager(self.mcio_dir, save=True) as cm:
            cm.config.instances.pop(instance_name)
