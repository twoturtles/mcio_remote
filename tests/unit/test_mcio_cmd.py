import shutil
from pathlib import Path

import minecraft_launcher_lib
import pytest

import mcio_remote as mcio
from mcio_remote.scripts import mcio_cmd

INST_NAME = "test-instance"


@pytest.fixture
def test_config(tmp_path: Path) -> Path:
    """Write test config to tmp_path dir. Returns tmp mcio_dir."""
    with mcio.config.ConfigManager(tmp_path, save=True) as cm:
        cm.config.instances[INST_NAME] = mcio.config.InstanceConfig(
            name=INST_NAME,
            launch_version="test-launch-version",
            minecraft_version="test-mc-version",
        )
    return tmp_path


@pytest.fixture
def test_mcio_dir(fixtures_dir: Path, tmp_path: Path) -> Path:
    shutil.copytree(fixtures_dir / "test_mcio_dir", tmp_path, dirs_exist_ok=True)
    return tmp_path


def test_world_instance_delete(test_mcio_dir: Path) -> None:
    world_path = test_mcio_dir / "instances/test_inst1/saves/World1"
    instance_name = "test_inst1"
    world_name = "World1"

    def world_exists_in_config() -> bool:
        with mcio.config.ConfigManager(mcio_dir=test_mcio_dir) as cm:
            return world_name in cm.config.instances[instance_name].worlds

    assert world_path.exists()
    assert world_exists_in_config()

    wm = mcio.world.WorldManager(mcio_dir=test_mcio_dir)
    wm.delete_cmd(f"{instance_name}:{world_name}")

    assert not world_path.exists()
    assert not world_exists_in_config()


def test_world_storage_delete(test_mcio_dir: Path) -> None:
    world_path = test_mcio_dir / "world_storage/World1"
    world_name = "World1"

    def world_exists_in_config() -> bool:
        with mcio.config.ConfigManager(mcio_dir=test_mcio_dir) as cm:
            return world_name in cm.config.world_storage

    assert world_path.exists()
    assert world_exists_in_config()

    wm = mcio.world.WorldManager(mcio_dir=test_mcio_dir)
    wm.delete_cmd(f"storage:{world_name}")

    assert not world_path.exists()
    assert not world_exists_in_config()


def test_instance_launch_list(
    test_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mock the arguments
    mcio_dir = test_config
    monkeypatch.setattr(
        "sys.argv",
        [
            "mcio",  # Program name
            "inst",  # Main command
            "launch",  # Subcommand
            INST_NAME,
            "--list",  # Show command list
            "--mcio-dir",
            str(mcio_dir),  # Use temp directory
        ],
    )

    command = ["foo", "--userType", "bar"]
    monkeypatch.setattr(
        minecraft_launcher_lib.command,
        "get_minecraft_command",
        lambda *args, **kwargs: command,
    )

    # Get the parsed arguments
    args, _ = mcio_cmd.base_parse_args()

    # Create and run the launch command
    cmd = mcio_cmd.InstanceLaunchCmd()
    cmd.run(args)

    # Not actually checking anything. Just want the code to run.
