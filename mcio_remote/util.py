import argparse
import logging
import queue
import shutil
import time
import types
from pathlib import Path
from typing import Any, Literal, TypeVar

import minecraft_launcher_lib as mll
import requests
from tqdm import tqdm

LOG = logging.getLogger(__name__)


COLORS = {
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "red-background": "\033[41m",
    "reset": "\033[0m",
}


def logging_add_arg(
    parser: argparse.ArgumentParser, default: int | str = "INFO"
) -> None:
    """Add a default logging argument to argparse"""
    parser.add_argument(
        "--log-level",
        "-L",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=default,
        help="Set the logging level (default: INFO)",
    )


def logging_init(
    *,
    args: argparse.Namespace | None = None,
    level: int | str | None = None,
    color: str | None = "cyan",
) -> None:
    """Default log init. If args are passed (see logging_add_arg), level is pulled
    from that. Otherwise uses a passed in level. Finally defaults to INFO"""
    if color is not None:
        color = COLORS[color]
        reset = COLORS["reset"]
    else:
        color = reset = ""

    if args is not None:
        level = getattr(logging, args.log_level.upper(), logging.INFO)
    elif level is None:
        level = logging.INFO

    fmt = f"{color}[%(asctime)s] [%(threadName)s/%(levelname)s] (%(name)s) %(message)s{reset}"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


# For LatestItemQueue
T = TypeVar("T")


class LatestItemQueue(queue.Queue[T]):
    """
    Threadsafe Queue that only saves the most recent item.
    Puts replace any item on the queue.
    """

    def __init__(self) -> None:
        super().__init__(maxsize=1)

    def put(self, item: T) -> bool:  # type: ignore[override]
        """Return True if the previous packet had to be dropped"""
        dropped = False
        try:
            # Discard the current item if the queue isn't empty
            self.get_nowait()
            dropped = True
        except queue.Empty:
            pass

        super().put(item)
        return dropped

    def get(self, block: bool = True, timeout: float | None = None) -> T:
        """
        The same as Queue.get, except this automatically calls task_done()
        I'm not sure task_done() really matters for how we're using Queue.
        Can raise queue.Empty if non-blocking or timeout
        """
        item = super().get(block=block, timeout=timeout)
        super().task_done()
        return item


class TrackPerSecond:
    def __init__(self, name: str, log_time: float | None = 10.0):
        self.name = name
        self.start = time.time()
        self.end = self.start
        self.item_count = 0

        self.log_start = self.start
        self.log_time = log_time
        self.log_count = 0

    def count(self) -> None:
        """Increment the counter and log every log_time"""
        self.end = time.time()
        self.item_count += 1
        self.log_count += 1
        if self.log_time is not None and self.end - self.log_start >= self.log_time:
            per_sec = self.log_count / (self.end - self.log_start)
            LOG.info(f"{self.name}: {per_sec:.1f}")
            self.log_count = 0
            self.log_start = self.end

    def avg_rate(self) -> float:
        """Return the average rate"""
        return self.item_count / (self.end - self.start)


class OptionsTxt:
    """Load/Save options.txt. Keeps everything as strings.
    To work with server.properties instead, set separator to "="
    """

    def __init__(
        self,
        options_path: Path | str,
        separator: Literal[":", "="] = ":",
        save: bool = False,
    ) -> None:
        """Set save to true to save automatically on exiting"""
        self.save_on_exit = save
        self.path = Path(options_path).expanduser()
        self.sep = separator
        self.options: dict[str, str] | None = None

    def load(self) -> None:
        """Load options from file."""
        if not self.path.exists():
            # XXX Should we let the user know instead of creating an empty options?
            self.options = {}
            return

        with self.path.open("r") as f:
            txt = f.read()
        lines = txt.strip().split("\n")
        self.options = {}
        for line in lines:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            key, value = line.split(self.sep, 1)
            key = key.strip()
            value = value.strip()
            self.options[key] = value

    def save(self) -> None:
        """Save options back to file"""
        assert self.options is not None
        with self.path.open("w") as f:
            for key, value in self.options.items():
                f.write(f"{key}{self.sep}{value}\n")

    def clear(self) -> None:
        """Clear the file"""
        self.options = {}

    def __getitem__(self, key: str) -> str:
        assert self.options is not None
        return self.options[key]

    def __setitem__(self, key: str, value: str) -> None:
        assert self.options is not None
        self.options[key] = value

    def __enter__(self) -> "OptionsTxt":
        self.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None:
        if exc_type is None:
            # Clean exit
            if self.save_on_exit:
                self.save()
        return None


class InstallProgress:
    """Progress bar for minecraft_launcher_lib installer"""

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


##
# Mojang web API utils
def mojang_get_version_manifest() -> dict[Any, Any]:
    """Example:
    {
      "latest": {
        "release": "1.21.4",
        "snapshot": "1.21.4"
      },
      "versions": [
        {
          "id": "1.21.4",
          "type": "release",
          "url": "https://piston-meta.mojang.com/v1/packages/a3bcba436caa849622fd7e1e5b89489ed6c9ac63/1.21.4.json",
          "time": "2024-12-03T10:24:48+00:00",
          "releaseTime": "2024-12-03T10:12:57+00:00",
          "sha1": "a3bcba436caa849622fd7e1e5b89489ed6c9ac63",
          "complianceLevel": 1
        },
    """
    versions_url = "https://launchermeta.mojang.com/mc/game/version_manifest_v2.json"
    response = requests.get(versions_url)
    response.raise_for_status()
    manifest: dict[Any, Any] = response.json()
    return manifest


def mojang_get_version_info(mc_version: str) -> dict[str, Any]:
    manifest = mojang_get_version_manifest()
    ver_list = manifest["versions"]
    ver_info: dict[str, Any]
    for ver_info in ver_list:
        if ver_info["id"] == mc_version:
            return ver_info
    raise ValueError(f"Version not found: {mc_version}")


def mojang_get_version_details(mc_version: str) -> dict[str, Any]:
    ver_info = mojang_get_version_info(mc_version)
    ver_details_url = ver_info["url"]

    response = requests.get(ver_details_url)
    response.raise_for_status()
    ver_details: dict[str, Any] = response.json()
    return ver_details


##
# Misc utils


def rmrf(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def copy_dir(src: Path, dst: Path, overwrite: bool = False) -> None:
    if not src.exists():
        raise ValueError(f"Source is missing: {src}")
    if not src.is_dir():
        raise ValueError(f"Source is not a directory: {src}")
    if dst.exists():
        if overwrite:
            rmrf(dst)
        else:
            raise ValueError(f"Destination exists: {dst}")
    shutil.copytree(src, dst)
