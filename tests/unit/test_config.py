from pathlib import Path
from typing import Generator

import pytest

from mcio_remote import config


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Generator[Path, None, None]:
    config_dir = tmp_path / ".mcio"
    config_dir.mkdir()
    yield config_dir


def test_config_basic(fixtures_dir: Path) -> None:
    with config.ConfigManager(mcio_dir=fixtures_dir) as cm:
        cfg = cm.config
        assert "Inst1" in cfg.instances
        assert "Inst2" in cfg.instances
        assert cfg.instances["Inst2"].name == "Inst2"
        assert cfg.instances["Inst2"].worlds["World1"].seed == "duff"


def test_config_from_dict() -> None:
    test_data = {
        "config_version": 0,
        "instances": {
            "test_instance": {
                "name": "test_instance",
                "launch_version": "fabric-loader-0.16.9-1.21.3",
                "minecraft_version": "1.21.3",
                "worlds": {},
            }
        },
        "world_storage": {},
        "servers": {},
    }
    cfg = config.Config.from_dict(test_data)
    assert cfg is not None
    assert cfg.instances["test_instance"].name == "test_instance"
    assert cfg.instances["test_instance"].minecraft_version == "1.21.3"


def test_config_manager_save_load(fixtures_dir: Path, temp_config_file: Path) -> None:
    with config.ConfigManager(mcio_dir=fixtures_dir) as cm:
        cfg = cm.config
    with config.ConfigManager(temp_config_file, save=True) as cm:
        cm.config = cfg
    with config.ConfigManager(temp_config_file) as cm:
        loaded_config = cm.config
        assert loaded_config.instances["Inst1"].name == "Inst1"
        assert loaded_config.instances["Inst1"].minecraft_version == "1.21.3"


def test_invalid_config() -> None:
    invalid_data = {
        "config_version": "0",  # Should be int
        "instances": [],  # Should be dict
    }
    assert config.Config.from_dict(invalid_data) is None
