from typing import Any

from mcio_remote.types import GlfwAction, InputEvent, InputID


class DegreesToPixels:
    """Convert a change in degrees of pitch and yaw to absolute pixels.
    This class allows an environment to use degrees of change of pitch and yaw and converts them into
    the equivalent absolute pixels for input into Minecraft.

    Minecraft takes in cursor position in terms of pixel coordinates and
    translates that into degrees of change in the viewpoint. When the mouse
    sensitivity is set to the default (0.5), the conversion is simply 1 pixel per 0.15
    degrees. This class is using that constant scaling factor.

    See Mouse.updateMouse() and Entity.changeLookDirection() in Minecraft (yarn mappings) for details.
    """

    DEGREES_PER_PIXEL = 0.15
    PIXELS_PER_DEGREE = 1 / DEGREES_PER_PIXEL

    def __init__(self) -> None:
        self.x: int = 0
        self.y: int = 0

    def update(self, *, yaw_delta: float, pitch_delta: float) -> tuple[int, int]:
        """Delta arguments are in degrees. Returns the new cursor position in pixels."""
        self.x += int(yaw_delta * self.PIXELS_PER_DEGREE)
        self.y += int(pitch_delta * self.PIXELS_PER_DEGREE)
        return self.x, self.y


class InputStateManager:
    def __init__(self) -> None:
        """Tracks the state of input keys and mouse buttons. This converts a
        stream of actions to input events. Consecutive actions of the same type
        only generate a single event. E.g., if two Press actions are generated
        in a row, only the first Press action will generate a Press event.
        With this, consecutive Presses are treated as a Press and Hold.
        """
        self.pressed_set: set[InputID] = set()

    def update(self, pressed: set[InputID], released: set[InputID]) -> list[InputEvent]:
        """Return set of updates to send to Minecraft. Also updates pressed_set."""
        update_events: list[InputEvent] = []

        new_presses = pressed - self.pressed_set
        for input_id in new_presses:
            update_events.append(InputEvent.from_id(input_id, GlfwAction.PRESS))

        new_releases = self.pressed_set & released
        for input_id in new_releases:
            update_events.append(InputEvent.from_id(input_id, GlfwAction.RELEASE))

        self.pressed_set |= new_presses
        self.pressed_set -= new_releases

        return update_events

    def process_action(
        self, action: dict[str, Any], input_map: dict[str, InputID]
    ) -> list[InputEvent]:
        """Prepare an action to be passed to update - Build sets of InputIDs based on Pressed or Not.
        Pass those sets to update and return the result.
        action - instance of an environment action. The keys are action names.
        input_map - maps action names to InputIDs.
        """
        # Build sets of which InputIDs are pressed or not
        pressed_set: set[InputID] = set()
        released_set: set[InputID] = set()
        for action_name, action_val in action.items():
            if action_name not in input_map:
                # The action will contain non-key/button fields
                continue
            input_id = input_map[action_name]
            # action_val is Discrete(2), so either np.int64(0) or np.int64(1)
            if bool(action_val):
                pressed_set.add(input_id)
            else:
                released_set.add(input_id)

        return self.update(pressed_set, released_set)
