#
# Example allowing human control through MCio
#

import queue
import time
from typing import Any

import glfw  # type: ignore

from . import logger
from . import gui
from . import controller
from . import network
from . import util

LOG = logger.LOG.get_logger(__name__)


class MCioGUI:
    def __init__(self, name: str = "MCio GUI", scale: float = 1.0, fps: int = 60):
        self.scale = scale
        self.fps = fps if fps > 0 else 60
        self.running = True
        self.gui = gui.ImageStreamGui("MCio GUI", scale=scale, width=800, height=600)
        self.controller = controller.ControllerAsync()

        # Set callbacks. Defaults are good enough for resize and focus.
        self.gui.set_callbacks(
            key_callback=self.key_callback,
            cursor_position_callback=self.cursor_position_callback,
            mouse_button_callback=self.mouse_button_callback,
        )

    def key_callback(
        self, window: Any, key: int, scancode: int, action: int, mods: int
    ) -> None:
        """Handle keyboard input"""
        if key == glfw.KEY_Q and action == glfw.PRESS:
            # Quit handling
            self.running = False
            return
        if action == glfw.REPEAT:
            # Skip action REPEAT.
            return

        # Pass everything else to Minecraft
        action_pkt = network.ActionPacket(keys=[(key, action)])
        self.controller.send_action(action_pkt)

    def cursor_position_callback(self, window: Any, xpos: float, ypos: float) -> None:
        """Handle mouse movement. Only watch the mouse when we're focused."""
        if self.gui.is_focused:
            # If we're scaling the window, also scale the position so things line up
            # XXX If the user manually resizes the window, the scaling goes out of whack.
            # Need to change the scale based on actual window size vs frame size
            scaled_pos = (int(xpos / self.scale), int(ypos / self.scale))
            action = network.ActionPacket(cursor_pos=[scaled_pos])
            self.controller.send_action(action)

    def mouse_button_callback(
        self, window: Any, button: int, action: int, mods: int
    ) -> None:
        """Handle mouse button events"""
        action_pkt = network.ActionPacket(mouse_buttons=[(button, action)])
        self.controller.send_action(action_pkt)

    def show(self, observation: network.ObservationPacket) -> None:
        """Show frame to the user"""
        if observation.frame_png:
            # Link cursor mode to Minecraft.
            self.gui.set_cursor_mode(observation.cursor_mode)
            frame = observation.get_frame_with_cursor()
            self.gui.show(frame)

    def run(self) -> None:
        """Main application loop
        NOTE: This must run on the main thread on MacOS
        """
        frame_time = 1.0 / self.fps
        fps_track = util.TrackPerSecond("FPS")
        while self.running:
            frame_start = time.perf_counter()
            try:
                observation = self.controller.recv_observation(block=False)
            except queue.Empty:
                pass
            else:
                LOG.debug(observation)
                self.show(observation)

            # Calculate sleep time to maintain target FPS
            elapsed = time.perf_counter() - frame_start
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            fps_track.count()

        # Cleanup
        LOG.info("Exiting...")
        self.close()

    def close(self) -> None:
        """Clean up resources"""
        self.controller.close()
        self.gui.close()
