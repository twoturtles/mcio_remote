from unittest.mock import MagicMock

import numpy as np
import pytest

from mcio_remote import network, types
from mcio_remote.envs import mcio_env


@pytest.fixture
def default_mcio_env() -> mcio_env.MCioEnv:
    return mcio_env.MCioEnv(types.RunOptions())


@pytest.fixture
def action_space_sample1() -> mcio_env.MCioAction:
    return {
        "cursor_pos_rel": np.array([827.648, 22.274418], dtype=np.float32),
        "keys": {"A": 0, "D": 0, "E": 1, "L_SHIFT": 1, "S": 1, "SPACE": 1, "W": 1},
        "mouse_buttons": {"LEFT": 1, "RIGHT": 0},
    }


@pytest.fixture
def mock_controller(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    mock_ctrl_sync = MagicMock()
    mock_ctrl_async = MagicMock()
    monkeypatch.setattr("mcio_remote.controller.ControllerSync", mock_ctrl_sync)
    monkeypatch.setattr("mcio_remote.controller.ControllerAsync", mock_ctrl_async)

    # Return objects that tests might need to access
    return {
        "ctrl_sync": mock_ctrl_sync,
        "ctrl_async": mock_ctrl_async,
    }


def test_action_fixture_is_valid(
    default_mcio_env: mcio_env.MCioEnv, action_space_sample1: mcio_env.MCioAction
) -> None:
    assert action_space_sample1 in default_mcio_env.action_space


# Smoke test of env
def test_env_smoke(
    mock_controller: dict[str, MagicMock], action_space_sample1: mcio_env.MCioAction
) -> None:
    env = mcio_env.MCioEnv(
        types.RunOptions(mcio_mode=types.MCioMode.SYNC), launch=False
    )
    obs, info = env.reset()
    assert isinstance(obs, dict)  # mcio_env.MCioObservation
    assert "frame" in obs
    env.step(action_space_sample1)
    with pytest.raises(ValueError):
        env.step({"Invalid Action": "will fail"})


def test_step_assert(
    mock_controller: dict[str, MagicMock], action_space_sample1: mcio_env.MCioAction
) -> None:
    env = mcio_env.MCioEnv(
        types.RunOptions(mcio_mode=types.MCioMode.SYNC), launch=False
    )
    with pytest.raises(AssertionError):
        env.step(action_space_sample1)  # No controller because reset hasn't been called


def test_env_with_commands(
    mock_controller: dict[str, MagicMock], action_space_sample1: mcio_env.MCioAction
) -> None:
    mock_ctrl_class: MagicMock = mock_controller["ctrl_sync"]
    env = mcio_env.MCioEnv(
        types.RunOptions(mcio_mode=types.MCioMode.SYNC), launch=False
    )
    cmds = ["command one", "command two"]

    def _check_send_action(send_action_mock: MagicMock) -> None:
        assert send_action_mock.call_count == 1
        action = send_action_mock.call_args.args[0]
        assert isinstance(action, network.ActionPacket)
        assert action.commands == cmds
        send_action_mock.reset_mock()

    # Check commands through reset
    env.reset(options={"commands": cmds})
    send_action = mock_ctrl_class.return_value.send_action
    _check_send_action(send_action)

    env.step(action_space_sample1, options={"commands": cmds})
    _check_send_action(send_action)


def test_action_to_packet(
    default_mcio_env: mcio_env.MCioEnv, action_space_sample1: mcio_env.MCioAction
) -> None:
    inputs = [
        types.InputEvent.from_ints(*ev)
        # type, code, action
        for ev in [
            (0, 32, 1),
            (0, 69, 1),
            (0, 83, 1),
            (0, 87, 1),
            (0, 340, 1),
            (1, 0, 1),
        ]
    ]
    expected1 = network.ActionPacket(
        version=network.MCIO_PROTOCOL_VERSION,
        sequence=0,
        commands=[],
        stop=False,
        inputs=inputs,
        cursor_pos=[(827, 22)],
    )
    pkt = default_mcio_env._action_to_packet(action_space_sample1)

    # Sort to ensure reproducible order
    expected1.inputs.sort()
    pkt.inputs.sort()
    assert pkt == expected1

    expected2 = network.ActionPacket(
        version=network.MCIO_PROTOCOL_VERSION,
        sequence=0,
        commands=[],
        inputs=[],
        cursor_pos=[(827, 22)],
    )
    # Passing the same action. Keys and mouse_buttons should be cleared since they're already set.
    pkt = default_mcio_env._action_to_packet(action_space_sample1)
    assert pkt == expected2
