import pickle
from pathlib import Path
from typing import Any

import pytest

from mcio_remote import types
from mcio_remote.envs import minerl_env


@pytest.fixture
def minerl_sample() -> Any:
    """The pkl contains a dict with an action and an observation from Minerl 1.0.
    The frame size is 640x360."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    with open(fixtures_dir / "minerl_sample.pkl", "rb") as f:
        sample = pickle.load(f)
    return sample


@pytest.fixture
def default_minerl_env() -> minerl_env.MinerlEnv:
    """Match the default minerl env"""
    args = minerl_env.MinerlEnvArgs(types.RunOptions(width=640, height=360))
    return minerl_env.MinerlEnv(args)


def test_minerl_is_valid(
    default_minerl_env: minerl_env.MinerlEnv, minerl_sample: Any
) -> None:
    """Check that at least the sample from minerl is valid in the mcio minerl env."""
    assert minerl_sample["action"] in default_minerl_env.action_space
    assert minerl_sample["observation"] in default_minerl_env.observation_space
