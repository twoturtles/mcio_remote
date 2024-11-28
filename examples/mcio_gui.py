#
# Example allowing human control through MCio
#

import queue
import argparse
import textwrap

import glfw
import OpenGL.GL as gl

import numpy as np

import mcio_remote as mcio
from mcio_remote import LOG

class MCioGUI:
    def __init__(self, name="MCio GUI", scale=1.0):
        self.scale = scale
        self.running = True
        self.gui = mcio.ImageStreamGui("MCio GUI", scale=scale, width=800, height=600)
        self.controller = mcio.ControllerAsync()

        # Set callbacks. Defaults are good enough for resize and focus.
        self.gui.set_callbacks(
            key_callback=self.key_callback,
            cursor_position_callback=self.cursor_position_callback,
            mouse_button_callback=self.mouse_button_callback
        )
        
    def key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input"""
        if key == glfw.KEY_Q and action == glfw.PRESS:
            # Quit handling
            self.running = False
            return
        if action == glfw.REPEAT:
            # Skip action REPEAT.
            return

        # Pass everything else to Minecraft
        action = mcio.network.ActionPacket(keys=[(key, action)])
        self.controller.send_action(action)

    def cursor_position_callback(self, window, xpos, ypos):
        """Handle mouse movement. Only watch the mouse when we're focused. """
        if self.gui.is_focused:
            # If we're scaling the window, also scale the position so things line up
            # XXX If the user manually resizes the window, the scaling goes out of whack.
            # Need to change the scale based on actual window size vs frame size
            scaled_pos = (xpos / self.scale, ypos / self.scale)
            action = mcio.network.ActionPacket(mouse_pos=[scaled_pos])
            self.controller.send_action(action)
        
    def mouse_button_callback(self, window, button, action, mods):
        """Handle mouse button events"""
        action = mcio.network.ActionPacket(mouse_buttons=[(button, action)])
        self.controller.send_action(action)

    def show(self, observation: mcio.network.ObservationPacket):
        '''Show frame to the user'''
        if observation.frame_png:
            # Link cursor mode to Minecraft.
            self.gui.set_cursor_mode(observation.cursor_mode)
            frame = observation.get_frame_with_cursor()
            self.gui.show(frame)
        
    def run(self):
        """Main application loop"""
        while self.running:
            try:
                observation = self.controller.recv_observation(block=False)
            except queue.Empty:
                pass
            else:
                LOG.debug(observation)
                self.show(observation)
            
        # Cleanup
        LOG.info("Exiting...")
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources"""
        self.controller.shutdown()
        self.gui.cleanup()

def parse_args():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''
            Provides a human GUI to MCio
            Q to quit
                                    '''),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Window scale factor')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app = MCioGUI(args.scale)
    app.run()