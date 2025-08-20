import mujoco as mj
from mujoco.glfw import glfw
import numpy as np

class MujocoRenderer:
    """
    A reusable class for rendering MuJoCo simulations using GLFW.
    """

    def __init__(self, model, data, height=900, width=1200, title="MuJoCo Simulation"):

        self.model = model
        self.data = data

        # For callback functions
        self._button_left = False
        self._button_middle = False
        self._button_right = False
        self._lastx = 0
        self._lasty = 0

        # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        self.window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # initialize visualization data structures
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # Set default camera configuration
        mj.mjv_defaultCamera(self.cam)

        # GLFW mouse and keyboard callbacks
        glfw.set_key_callback(self.window, self._keyboard_callback)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

    def _keyboard_callback(self, window, key, scancode, act, mods):
        """Handles keyboard button presses."""
        pass

    def _mouse_button_callback(self, window, button, act, mods):
        """Handles mouse button presses."""
        self._button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self._button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self._lastx, self._lasty = glfw.get_cursor_pos(window)

    def _mouse_move_callback(self, window, xpos, ypos):
        """Handles mouse movement for camera control."""
        dx = xpos - self._lastx
        dy = ypos - self._lasty
        self._lastx = xpos
        self._lasty = ypos

        if not (self._button_left or self._button_middle or self._button_right):
            return

        width, height = glfw.get_window_size(window)
        mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                     glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

        if self._button_right:
            action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self.model, action, dx / height, dy / height, self.scene, self.cam)

    def _scroll_callback(self, window, xoffset, yoffset):
        """Handles mouse scrolling for zooming."""
        mj.mjv_moveCamera(self.model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, self.scene, self.cam)

    def render(self):
        """Renders a single frame of the simulation."""
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        # Update scene and render
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)

        # Swap OpenGL buffers
        glfw.swap_buffers(self.window)

        # Process pending GUI events
        glfw.poll_events()

    def is_window_closed(self):
        """Checks if the rendering window should be closed."""
        return glfw.window_should_close(self.window)

    def close(self):
        """Cleans up and terminates GLFW."""
        glfw.terminate()
