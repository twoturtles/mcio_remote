import threading
import io
import queue

import glfw
import OpenGL.GL as gl

import numpy as np
from PIL import Image
import cv2

import mcio_remote as mcio

# Threads to handle zmq i/o.
class ControllerThreads:
    def __init__(self, host='localhost'):
        # Flag to signal threads to stop.
        self.running = threading.Event()
        self.running.set()

        self.action_queue = queue.Queue()
        self.state_queue = queue.Queue()

        self.mcio_conn = mcio.Connection()

        # Start threads
        self.state_thread = threading.Thread(target=self.state_thread_fn, name="StateThread")
        self.state_thread.daemon = True
        self.state_thread.start()

        self.action_thread = threading.Thread(target=self.action_thread_fn, name="ActionThread")
        self.action_thread.daemon = True
        self.action_thread.start()

    def action_thread_fn(self):
        print("ActionThread start")
        while self.running.is_set():
            action = self.action_queue.get()
            if action is None:
                break   # Action None to signal exit
            self.mcio_conn.send_action(action)
            self.action_queue.task_done()
        print("Action-Thread shut down")

    def state_thread_fn(self):
        print("StateThread start")
        while self.running.is_set():
            state = self.mcio_conn.recv_state()
            if state is None:
                continue    # Exiting or packet decode error
            self.state_queue.put(state)
        print("StateThread shut down")

    def shutdown(self):
        self.running.clear()
        self.mcio_conn.close()

        self.state_thread.join()
        # Send empty action to unblock ActionThread
        self.action_queue.put(None)
        self.action_thread.join()

class MCioGUI:
    def __init__(self, width=800, height=600):
        # Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed")
            
        # This fixes only filling the bottom-left 1/4 of the window on mac.
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.FALSE)
        # Create window
        self.window = glfw.create_window(width, height, "MCio GUI", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed")
            
        # Set up OpenGL context
        glfw.make_context_current(self.window)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        
        # Set callbacks
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_position_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_window_size_callback(self.window, self.resize_callback)

        self.controller = ControllerThreads()
        
    def key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input"""
        #print(f'Key={key} action={action}')
        
        # Quit handling
        if key == glfw.KEY_Q and action == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
            return

        if action == glfw.PRESS:
            action = mcio.network.ActionPacket(keys_pressed={key})
            self.controller.action_queue.put(action)
        elif action == glfw.RELEASE:
            action = mcio.network.ActionPacket(keys_released={key})
            self.controller.action_queue.put(action)
        # Skip action REPEAT.

    # XXX When the cursor gets to the edge of the screen you turn any farther because the
    # cursor position doesn't change. Minecraft handles this, but doesn't show the cursor.
    def cursor_position_callback(self, window, xpos, ypos):
        """Handle mouse movement"""
        #print(f'Mouse {xpos} {ypos}')
        action = mcio.network.ActionPacket(mouse_pos_update=True, mouse_pos_x=xpos, mouse_pos_y=ypos)
        self.controller.action_queue.put(action)
        
    def mouse_button_callback(self, window, button, action, mods):
        """Handle mouse button events"""
        if action == glfw.PRESS:
            action = mcio.network.ActionPacket(mouse_buttons_pressed={button})
            self.controller.action_queue.put(action)
        elif action == glfw.RELEASE:
            action = mcio.network.ActionPacket(mouse_buttons_released={button})
            self.controller.action_queue.put(action)

    def resize_callback(self, window, width, height):
        """Handle window resize"""
        gl.glViewport(0, 0, width, height)
        # Force a redraw
        glfw.post_empty_event()
        
    def render(self, state: mcio.network.StatePacket):
        """Render graphics"""
        SCALE = 2
        
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        if state.frame_png:
            # Convert PNG bytes to image
            img = Image.open(io.BytesIO(state.frame_png))
            img_width, img_height = img.size
            
            # Scale the target dimensions
            target_width = img_width // SCALE
            target_height = img_height // SCALE
            
            # On first frame or if size changed, resize window
            current_width, current_height = glfw.get_window_size(self.window)
            if current_width != target_width or current_height != target_height:
                glfw.set_window_size(self.window, target_width, target_height)
            
            # Convert image to numpy array and flip vertically to pass to OpenGL
            img_data = np.array(img)
            img_data = np.flipud(img_data)
            
            # Scale the image data to fit the desired size
            img_data = cv2.resize(img_data, (target_width, target_height))
            
            # Create and bind texture
            texture = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            
            # Set texture parameters for scaling down/up
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            
            # Upload the scaled image to texture
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGB, 
                target_width, target_height, 0,
                gl.GL_RGB, gl.GL_UNSIGNED_BYTE, 
                img_data
            )
            
            # Enable texture mapping
            gl.glEnable(gl.GL_TEXTURE_2D)
            
            # Draw a quad (rectangle made of two triangles) that fills the screen
            # The quad goes from -1 to 1 in both x and y (OpenGL normalized coordinates)
            gl.glBegin(gl.GL_QUADS)
            # For each vertex, set texture coordinate (0-1) and vertex position (-1 to 1)
            gl.glTexCoord2f(0, 0); gl.glVertex2f(-1, -1)
            gl.glTexCoord2f(1, 0); gl.glVertex2f(1, -1)
            gl.glTexCoord2f(1, 1); gl.glVertex2f(1, 1)
            gl.glTexCoord2f(0, 1); gl.glVertex2f(-1, 1)
            gl.glEnd()
            
            # Clean up
            gl.glDisable(gl.GL_TEXTURE_2D)
            gl.glDeleteTextures([texture])
        
        glfw.swap_buffers(self.window)
        
    def run(self):
        """Main application loop"""
        while not glfw.window_should_close(self.window):
            # Poll for events
            glfw.poll_events()
            try:
                state = self.controller.state_queue.get(block=False)
                self.controller.state_queue.task_done()
            except queue.Empty:
                pass
            else:
                #print(state)
                self.render(state)
            
        # Cleanup
        print("Exiting...")
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources"""
        self.controller.shutdown()
        glfw.terminate()

if __name__ == "__main__":
    app = MCioGUI()
    app.run()