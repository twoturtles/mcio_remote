from unittest.mock import MagicMock

import glfw  # type: ignore
import numpy as np
import pytest

from mcio_remote import network, types
from mcio_remote.envs import mcio_env


@pytest.fixture
def default_mcio_env() -> mcio_env.MCioEnv:
    return mcio_env.MCioEnv(types.RunOptions())


@pytest.fixture
def action_space_sample1(default_mcio_env: mcio_env.MCioEnv) -> mcio_env.MCioAction:
    act = default_mcio_env.get_noop_action()
    act.update(
        {
            "cursor_delta": np.array([827, 22], dtype=np.int32),
            "A": 1,
            "W": 0,
            "LEFT_SHIFT": 1,
            "LEFT_BUTTON": 1,
        }
    )
    return act


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
    env = mcio_env.MCioEnv(types.RunOptions(mcio_mode=types.MCioMode.SYNC))
    obs, info = env.reset()
    assert isinstance(obs, dict)  # mcio_env.MCioObservation
    assert "frame" in obs
    env.step(action_space_sample1)


def test_step_assert(
    mock_controller: dict[str, MagicMock], action_space_sample1: mcio_env.MCioAction
) -> None:
    env = mcio_env.MCioEnv(types.RunOptions(mcio_mode=types.MCioMode.SYNC))
    with pytest.raises(AssertionError):
        env.step(action_space_sample1)  # No controller because reset hasn't been called


def test_env_with_commands(
    mock_controller: dict[str, MagicMock], action_space_sample1: mcio_env.MCioAction
) -> None:
    mock_ctrl_class: MagicMock = mock_controller["ctrl_sync"]
    env = mcio_env.MCioEnv(types.RunOptions(mcio_mode=types.MCioMode.SYNC))
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
        types.InputEvent(types.InputType.KEY, glfw.KEY_A, types.GlfwAction.PRESS),
        # W won't be in the intial action b/c keys start out released
        # types.InputEvent(types.InputType.KEY, glfw.KEY_W, types.GlfwAction.RELEASE),
        types.InputEvent(
            types.InputType.KEY, glfw.KEY_LEFT_SHIFT, types.GlfwAction.PRESS
        ),
        types.InputEvent(
            types.InputType.MOUSE, glfw.MOUSE_BUTTON_LEFT, types.GlfwAction.PRESS
        ),
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
    # Passing the same action. Keys and mouse_buttons should not be in the action since they're already set.
    pkt = default_mcio_env._action_to_packet(action_space_sample1)
    assert pkt == expected2
