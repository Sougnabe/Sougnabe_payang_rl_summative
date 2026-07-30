from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import pyglet
except ImportError:  # pragma: no cover - optional for headless runs
    pyglet = None

try:
    from OpenGL import GL as gl
    from OpenGL.GLU import gluPerspective, gluLookAt
except ImportError:  # pragma: no cover
    gl = None


@dataclass
class RenderState:
    agent_pos: np.ndarray
    base_pos: np.ndarray
    victim_positions: np.ndarray
    victim_pending: np.ndarray
    victim_discovered: np.ndarray
    hazard_centers: np.ndarray
    hazard_radii: np.ndarray
    battery: float
    kits_left: int
    step_count: int
    max_steps: int


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


class MissionRenderer:
    def __init__(self, world_size: float, width: int = 900, height: int = 700) -> None:
        self.world_size = world_size
        self.width = width
        self.height = height
        self._window: Any = None
        self._close_requested = False
        self._paused = False

        # Camera controls
        self._cam_azimuth = 45.0  # degrees
        self._cam_elevation = 35.0  # degrees
        self._cam_distance = 160.0
        self._cam_target = np.array([50.0, 50.0, 0.0], dtype=np.float32)
        self._dragging = False
        self._last_mouse = (0, 0)

        # Animation
        self._time = 0.0

    # ------------------------------------------------------------------
    #  Window / event handling
    # ------------------------------------------------------------------
    def on_key_press(self, symbol: int, modifiers: int) -> None:
        escape_keys = {27, ord("q"), ord("Q")}
        if pyglet is not None:
            escape_keys.add(getattr(pyglet.window.key, "ESCAPE", 27))
            escape_keys.add(getattr(pyglet.window.key, "Q", ord("q")))
        if symbol in escape_keys:
            self._close_requested = True
            return
        if symbol in {32, ord(" ")}:
            self._paused = not self._paused

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if pyglet is not None and button == pyglet.window.mouse.LEFT:
            self._dragging = True
            self._last_mouse = (x, y)

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        if pyglet is not None and button == pyglet.window.mouse.LEFT:
            self._dragging = False

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        if pyglet is not None and buttons & pyglet.window.mouse.LEFT:
            self._cam_azimuth += dx * 0.5
            self._cam_elevation = np.clip(self._cam_elevation - dy * 0.5, 5.0, 85.0)

    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self._cam_distance = np.clip(self._cam_distance - scroll_y * 8.0, 30.0, 400.0)

    # ------------------------------------------------------------------
    #  OpenGL helpers
    # ------------------------------------------------------------------
    def _setup_3d(self) -> None:
        gl.glViewport(0, 0, self.width, self.height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = self.width / self.height
        gluPerspective(45.0, aspect, 1.0, 800.0)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        # Camera position from spherical coords
        rad_az = math.radians(self._cam_azimuth)
        rad_el = math.radians(self._cam_elevation)
        cx = self._cam_target[0] + self._cam_distance * math.cos(rad_el) * math.sin(rad_az)
        cy = self._cam_target[1] + self._cam_distance * math.cos(rad_el) * math.cos(rad_az)
        cz = self._cam_target[2] + self._cam_distance * math.sin(rad_el)
        gluLookAt(cx, cy, cz, self._cam_target[0], self._cam_target[1], self._cam_target[2], 0.0, 0.0, 1.0)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_LINE_SMOOTH)

        # Lighting
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)

        light_pos = (150.0, 150.0, 200.0, 1.0)
        light_ambient = (0.3, 0.3, 0.3, 1.0)
        light_diffuse = (0.9, 0.9, 0.9, 1.0)
        light_specular = (0.4, 0.4, 0.4, 1.0)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_pos)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, light_ambient)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, light_diffuse)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, light_specular)

        gl.glEnable(gl.GL_LIGHT1)
        light1_pos = (-100.0, 50.0, 100.0, 1.0)
        light1_ambient = (0.15, 0.15, 0.2, 1.0)
        light1_diffuse = (0.4, 0.4, 0.5, 1.0)
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, light1_pos)
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_AMBIENT, light1_ambient)
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, light1_diffuse)

        gl.glShadeModel(gl.GL_SMOOTH)

    def _setup_2d_overlay(self) -> None:
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, self.width, 0, self.height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_LIGHTING)

    # ------------------------------------------------------------------
    #  Drawing primitives
    # ------------------------------------------------------------------
    def _draw_ground(self) -> None:
        """Draw a textured-looking ground plane with grid lines."""
        gl.glDisable(gl.GL_LIGHTING)
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.15, 0.25, 0.12, 1.0)
        gl.glVertex3f(0.0, 0.0, -0.5)
        gl.glVertex3f(self.world_size, 0.0, -0.5)
        gl.glVertex3f(self.world_size, self.world_size, -0.5)
        gl.glVertex3f(0.0, self.world_size, -0.5)
        gl.glEnd()

        # Grid lines
        gl.glColor4f(0.25, 0.40, 0.20, 0.4)
        gl.glBegin(gl.GL_LINES)
        step = 10.0
        for i in range(0, int(self.world_size) + 1, int(step)):
            x = float(i)
            gl.glVertex3f(x, 0.0, -0.4)
            gl.glVertex3f(x, self.world_size, -0.4)
            gl.glVertex3f(0.0, x, -0.4)
            gl.glVertex3f(self.world_size, x, -0.4)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)

    def _draw_cylinder(self, x: float, y: float, radius: float, height: float, color: tuple[float, float, float], alpha: float = 1.0) -> None:
        """Draw a 3D cylinder at (x, y) with given radius and height."""
        slices = 24
        gl.glColor4f(color[0], color[1], color[2], alpha)
        # Side
        gl.glBegin(gl.GL_QUAD_STRIP)
        for i in range(slices + 1):
            angle = 2.0 * math.pi * i / slices
            dx = radius * math.cos(angle)
            dy = radius * math.sin(angle)
            gl.glNormal3f(math.cos(angle), math.sin(angle), 0.0)
            gl.glVertex3f(x + dx, y + dy, 0.0)
            gl.glVertex3f(x + dx, y + dy, height)
        gl.glEnd()
        # Top
        gl.glBegin(gl.GL_TRIANGLE_FAN)
        gl.glNormal3f(0.0, 0.0, 1.0)
        gl.glVertex3f(x, y, height)
        for i in range(slices + 1):
            angle = 2.0 * math.pi * i / slices
            gl.glVertex3f(x + radius * math.cos(angle), y + radius * math.sin(angle), height)
        gl.glEnd()

    def _draw_sphere(self, x: float, y: float, z: float, radius: float, color: tuple[float, float, float], alpha: float = 1.0) -> None:
        """Draw a UV-sphere."""
        stacks = 12
        slices = 16
        gl.glColor4f(color[0], color[1], color[2], alpha)
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + float(i) / stacks)
            z0 = z + radius * math.sin(lat0)
            zr0 = radius * math.cos(lat0)
            lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
            z1 = z + radius * math.sin(lat1)
            zr1 = radius * math.cos(lat1)
            gl.glBegin(gl.GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2.0 * math.pi * float(j) / slices
                nx = math.cos(lat1) * math.cos(lng)
                ny = math.cos(lat1) * math.sin(lng)
                nz = math.sin(lat1)
                gl.glNormal3f(nx, ny, nz)
                gl.glVertex3f(x + zr1 * math.cos(lng), y + zr1 * math.sin(lng), z1)
                nx = math.cos(lat0) * math.cos(lng)
                ny = math.cos(lat0) * math.sin(lng)
                nz = math.sin(lat0)
                gl.glNormal3f(nx, ny, nz)
                gl.glVertex3f(x + zr0 * math.cos(lng), y + zr0 * math.sin(lng), z0)
            gl.glEnd()

    def _draw_box(self, x: float, y: float, z: float, w: float, h: float, d: float, color: tuple[float, float, float], alpha: float = 1.0) -> None:
        """Draw a 3D box centered at (x, y, z)."""
        hw, hh, hd = w / 2, h / 2, d / 2
        vertices = [
            (-hw, -hh, -hd), (hw, -hh, -hd), (hw, hh, -hd), (-hw, hh, -hd),
            (-hw, -hh, hd), (hw, -hh, hd), (hw, hh, hd), (-hw, hh, hd),
        ]
        faces = [
            (0, 1, 2, 3, 0, 0, -1),  # bottom
            (4, 5, 6, 7, 0, 0, 1),   # top
            (0, 1, 5, 4, 0, -1, 0),  # front
            (2, 3, 7, 6, 0, 1, 0),   # back
            (0, 3, 7, 4, -1, 0, 0),  # left
            (1, 2, 6, 5, 1, 0, 0),   # right
        ]
        gl.glColor4f(color[0], color[1], color[2], alpha)
        gl.glBegin(gl.GL_QUADS)
        for face in faces:
            nx, ny, nz = face[4], face[5], face[6]
            gl.glNormal3f(nx, ny, nz)
            for idx in face[:4]:
                gl.glVertex3f(x + vertices[idx][0], y + vertices[idx][1], z + vertices[idx][2])
        gl.glEnd()

    def _draw_drone(self, x: float, y: float, z: float, color: tuple[float, float, float]) -> None:
        """Draw a simple quadcopter drone."""
        body_w, body_h, body_d = 4.0, 1.5, 4.0
        arm_len = 5.0
        rotor_r = 2.5

        # Body
        self._draw_box(x, y, z + body_h / 2, body_w, body_h, body_d, color)

        # Arms (4 diagonal arms)
        arm_color = (0.4, 0.4, 0.4)
        arm_h = 0.4
        for angle in [45, 135, 225, 315]:
            rad = math.radians(angle)
            ax = x + arm_len * math.cos(rad)
            ay = y + arm_len * math.sin(rad)
            # Draw arm as thin box
            mx, my = (x + ax) / 2, (y + ay) / 2
            self._draw_box(mx, my, z + arm_h / 2, arm_len * 0.7, arm_h, 0.5, arm_color)

            # Rotor disc (flat cylinder)
            gl.glDisable(gl.GL_LIGHTING)
            rotor_color = (0.7, 0.7, 0.8, 0.35)
            gl.glColor4f(*rotor_color)
            gl.glBegin(gl.GL_TRIANGLE_FAN)
            gl.glVertex3f(ax, ay, z + 1.2)
            for i in range(21):
                a = 2.0 * math.pi * i / 20
                gl.glVertex3f(ax + rotor_r * math.cos(a), ay + rotor_r * math.sin(a), z + 1.2)
            gl.glEnd()
            gl.glEnable(gl.GL_LIGHTING)

        # Center light
        gl.glDisable(gl.GL_LIGHTING)
        gl.glColor4f(1.0, 0.2, 0.2, 1.0)
        gl.glBegin(gl.GL_TRIANGLE_FAN)
        gl.glVertex3f(x, y, z + body_h + 0.5)
        for i in range(21):
            a = 2.0 * math.pi * i / 20
            gl.glVertex3f(x + 0.8 * math.cos(a), y + 0.8 * math.sin(a), z + body_h + 0.5)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)

    def _draw_base_station(self, x: float, y: float) -> None:
        """Draw a base station with a landing pad."""
        # Pad
        self._draw_cylinder(x, y, 10.0, 0.5, (0.3, 0.5, 0.8))
        # Tower
        self._draw_cylinder(x, y, 2.0, 6.0, (0.2, 0.4, 0.7))
        # Antenna ball
        self._draw_sphere(x, y, 7.0, 1.5, (0.6, 0.8, 1.0))
        # Ring
        gl.glDisable(gl.GL_LIGHTING)
        gl.glColor4f(0.4, 0.7, 1.0, 0.3)
        gl.glBegin(gl.GL_LINE_LOOP)
        for i in range(36):
            a = 2.0 * math.pi * i / 36
            gl.glVertex3f(x + 12.0 * math.cos(a), y + 12.0 * math.sin(a), 0.0)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)

    def _draw_hazard(self, x: float, y: float, radius: float) -> None:
        """Draw a hazard zone as a toxic waste barrel / glowing area."""
        # Glow ring on ground
        gl.glDisable(gl.GL_LIGHTING)
        gl.glColor4f(0.8, 0.15, 0.15, 0.15)
        gl.glBegin(gl.GL_TRIANGLE_FAN)
        gl.glVertex3f(x, y, -0.3)
        for i in range(37):
            a = 2.0 * math.pi * i / 36
            gl.glVertex3f(x + radius * math.cos(a), y + radius * math.sin(a), -0.3)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)

        # Barrels / pillars around the edge
        n_barrels = max(6, int(radius * 0.8))
        for i in range(n_barrels):
            a = 2.0 * math.pi * i / n_barrels
            bx = x + (radius - 1.5) * math.cos(a)
            by = y + (radius - 1.5) * math.sin(a)
            self._draw_cylinder(bx, by, 1.2, 3.0, (0.6, 0.1, 0.1))
            # Warning stripe
            self._draw_cylinder(bx, by, 1.3, 0.3, (0.9, 0.8, 0.1))

        # Center glow
        gl.glDisable(gl.GL_LIGHTING)
        gl.glColor4f(0.9, 0.2, 0.1, 0.08)
        gl.glBegin(gl.GL_TRIANGLE_FAN)
        gl.glVertex3f(x, y, -0.2)
        for i in range(37):
            a = 2.0 * math.pi * i / 36
            gl.glVertex3f(x + radius * 0.6 * math.cos(a), y + radius * 0.6 * math.sin(a), -0.2)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)

    def _draw_victim(self, x: float, y: float, discovered: bool, pending: bool) -> None:
        """Draw a victim as a person figure."""
        if pending and discovered:
            color = (0.96, 0.82, 0.25)  # Gold - discovered but needs delivery
        elif pending and not discovered:
            color = (0.5, 0.5, 0.5)  # Gray - undiscovered
        else:
            color = (0.37, 0.82, 0.47)  # Green - delivered

        # Body (sphere)
        self._draw_sphere(x, y, 1.5, 2.0, color)
        # Head
        self._draw_sphere(x, y, 4.5, 1.2, (0.9, 0.85, 0.8))
        # Status indicator
        if pending:
            # Floating marker
            gl.glDisable(gl.GL_LIGHTING)
            marker_color = (1.0, 0.3, 0.1, 0.6) if discovered else (0.5, 0.5, 0.5, 0.4)
            gl.glColor4f(*marker_color)
            gl.glBegin(gl.GL_TRIANGLE_FAN)
            gl.glVertex3f(x, y, 6.5)
            for i in range(21):
                a = 2.0 * math.pi * i / 20
                gl.glVertex3f(x + 1.5 * math.cos(a), y + 1.5 * math.sin(a), 6.5)
            gl.glEnd()
            gl.glEnable(gl.GL_LIGHTING)

    def _draw_battery_bar(self, battery: float) -> None:
        """Draw a 3D-style battery bar in the overlay."""
        bar_w = 260
        bar_h = 22
        x0 = 20
        y0 = self.height - 50

        # Background
        gl.glColor4f(0.15, 0.15, 0.15, 0.85)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(x0 - 2, y0 - 2)
        gl.glVertex2f(x0 + bar_w + 2, y0 - 2)
        gl.glVertex2f(x0 + bar_w + 2, y0 + bar_h + 2)
        gl.glVertex2f(x0 - 2, y0 + bar_h + 2)
        gl.glEnd()

        # Fill
        fill = int(bar_w * float(np.clip(battery, 0.0, 1.0)))
        if battery > 0.5:
            col = (0.24, 0.82, 0.47)
        elif battery > 0.25:
            col = (0.9, 0.7, 0.15)
        else:
            col = (0.85, 0.2, 0.15)
        gl.glColor4f(col[0], col[1], col[2], 0.9)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(x0, y0)
        gl.glVertex2f(x0 + fill, y0)
        gl.glVertex2f(x0 + fill, y0 + bar_h)
        gl.glVertex2f(x0, y0 + bar_h)
        gl.glEnd()

        # Border
        gl.glColor4f(0.6, 0.6, 0.6, 0.9)
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex2f(x0, y0)
        gl.glVertex2f(x0 + bar_w, y0)
        gl.glVertex2f(x0 + bar_w, y0 + bar_h)
        gl.glVertex2f(x0, y0 + bar_h)
        gl.glEnd()

    def _draw_text_overlay(self, text: str, x: int, y: int, color: tuple[float, float, float] = (1.0, 1.0, 1.0), scale: int = 1) -> None:
        """Simple bitmap text overlay."""
        font_map = {
            "0": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
            "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
            "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
            "3": ["01110", "10001", "00001", "00110", "00001", "10001", "01110"],
            "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
            "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
            "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
            "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
            "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
            "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
            ".": ["00000", "00000", "00000", "00000", "00000", "00100", "00100"],
            "%": ["10001", "10010", "00100", "01000", "10000", "01001", "00010"],
            ":": ["00000", "00100", "00000", "00000", "00100", "00000", "00000"],
            "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
            "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
            "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
            "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
            "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
            "Y": ["10001", "10001", "10001", "01110", "00100", "00100", "00100"],
            "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
            "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
            "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
            "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
            "V": ["10001", "10001", "10001", "10001", "10001", "01110", "00100"],
            "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
            "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
            "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
            "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
            "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
            "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
            "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
            "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
            "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
            "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
            "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
            "X": ["10001", "10001", "01110", "00100", "01110", "10001", "10001"],
            "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
            "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
            "J": ["00111", "00001", "00001", "00001", "00001", "10001", "01110"],
            "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
            "+": ["00000", "00000", "00100", "01110", "00100", "00000", "00000"],
            "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
            " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
        }
        cursor_x = x
        for ch in text.upper():
            glyph = font_map.get(ch)
            if glyph is None:
                glyph = font_map[" "]
            for row_idx, row in enumerate(glyph):
                for col_idx, bit in enumerate(row):
                    if bit == "1":
                        px0 = cursor_x + col_idx * scale
                        py0 = y + row_idx * scale
                        gl.glColor4f(color[0], color[1], color[2], 0.9)
                        gl.glBegin(gl.GL_QUADS)
                        gl.glVertex2f(px0, py0)
                        gl.glVertex2f(px0 + scale, py0)
                        gl.glVertex2f(px0 + scale, py0 + scale)
                        gl.glVertex2f(px0, py0 + scale)
                        gl.glEnd()
            cursor_x += (len(glyph[0]) + 1) * scale

    # ------------------------------------------------------------------
    #  Main render methods
    # ------------------------------------------------------------------
    def _create_window(self, visible: bool = False) -> Any:
        """Create a pyglet window with OpenGL compatibility profile."""
        try:
            # Try to create a compatibility profile context for fixed-function pipeline
            config = pyglet.gl.Config(
                major_version=2,
                minor_version=1,
                depth_size=24,
                double_buffer=True,
            )
            window = pyglet.window.Window(
                self.width, self.height,
                caption="Mission Environment (3D)",
                visible=visible,
                resizable=True,
                config=config,
            )
        except Exception:
            # Fall back to default config
            window = pyglet.window.Window(
                self.width, self.height,
                caption="Mission Environment (3D)",
                visible=visible,
                resizable=True,
            )
        window.push_handlers(self)
        return window

    def render_rgb(self, state: RenderState) -> np.ndarray:
        """Render to an offscreen RGB array for video capture."""
        if pyglet is None or gl is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Create offscreen context if needed
        if self._window is None:
            self._window = self._create_window(visible=False)

        # Make the offscreen context current
        self._window.switch_to()
        self._window.dispatch_events()

        self._render_frame(state)

        # Flush to ensure rendering completes
        gl.glFlush()
        gl.glFinish()

        # Read pixels
        buffer = (gl.GLubyte * (self.width * self.height * 3))(0)
        gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, buffer)
        img = np.frombuffer(buffer, dtype=np.uint8).reshape((self.height, self.width, 3))
        return np.ascontiguousarray(img)

    def _render_frame(self, state: RenderState) -> None:
        """Render a single 3D frame."""
        self._time += 0.05
        gl.glClearColor(0.08, 0.12, 0.18, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # ---- 3D scene ----
        self._setup_3d()

        # Update camera target to follow agent
        self._cam_target[0] = self._cam_target[0] * 0.92 + state.agent_pos[0] * 0.08
        self._cam_target[1] = self._cam_target[1] * 0.92 + state.agent_pos[1] * 0.08

        self._draw_ground()

        # Hazards
        for center, radius in zip(state.hazard_centers, state.hazard_radii):
            self._draw_hazard(float(center[0]), float(center[1]), float(radius))

        # Victims
        for idx, victim in enumerate(state.victim_positions):
            self._draw_victim(
                float(victim[0]), float(victim[1]),
                bool(state.victim_discovered[idx]),
                bool(state.victim_pending[idx]),
            )

        # Base station
        self._draw_base_station(float(state.base_pos[0]), float(state.base_pos[1]))

        # Agent drone (hover animation)
        hover_z = 1.5 + 0.5 * math.sin(self._time * 2.0)
        self._draw_drone(
            float(state.agent_pos[0]),
            float(state.agent_pos[1]),
            hover_z,
            (0.9, 0.9, 0.95),
        )

        # ---- 2D overlay ----
        self._setup_2d_overlay()

        self._draw_battery_bar(state.battery)

        info_lines = [
            f"Battery: {state.battery * 100:.1f}%",
            f"Kits: {state.kits_left} | Delivered: {sum(1 for p in state.victim_pending if not p)}/{len(state.victim_pending)}",
            f"Step: {state.step_count}/{state.max_steps}",
            "Space: Pause | Esc/Q: Quit | Mouse: Orbit/Zoom",
        ]
        for i, line in enumerate(info_lines):
            self._draw_text_overlay(line, 20, self.height - 80 - i * 22, (0.9, 0.9, 0.9), scale=1)

        # Legend
        legend_y = 20
        legend_items = [
            ("Drone", (0.9, 0.9, 0.95)),
            ("Base", (0.3, 0.5, 0.8)),
            ("Victim (undiscovered)", (0.5, 0.5, 0.5)),
            ("Victim (found)", (0.96, 0.82, 0.25)),
            ("Victim (rescued)", (0.37, 0.82, 0.47)),
            ("Hazard", (0.8, 0.15, 0.15)),
        ]
        for i, (label, col) in enumerate(legend_items):
            lx = self.width - 220
            ly = legend_y + i * 18
            gl.glColor4f(col[0], col[1], col[2], 0.8)
            gl.glBegin(gl.GL_QUADS)
            gl.glVertex2f(lx, ly)
            gl.glVertex2f(lx + 12, ly)
            gl.glVertex2f(lx + 12, ly + 12)
            gl.glVertex2f(lx, ly + 12)
            gl.glEnd()
            self._draw_text_overlay(label, lx + 18, ly, (0.8, 0.8, 0.8), scale=1)

        if self._paused:
            self._draw_text_overlay("PAUSED", self.width // 2 - 60, self.height // 2, (1.0, 0.8, 0.2), scale=3)

    def render_human(self, state: RenderState) -> bool:
        """Render to a visible pyglet window."""
        if pyglet is None or gl is None:
            return False

        if self._window is None:
            self._window = self._create_window(visible=True)

            @self._window.event
            def on_resize(w: int, h: int) -> None:
                self.width = max(100, w)
                self.height = max(100, h)
                gl.glViewport(0, 0, self.width, self.height)

        self._window.switch_to()
        self._window.dispatch_events()

        if getattr(self._window, "has_exit", False) or self._close_requested:
            self._window.close()
            self._window = None
            self._close_requested = False
            return False

        if not self._paused:
            self._render_frame(state)

        self._window.flip()
        return True

    def close(self) -> None:
        if self._window is not None:
            self._window.close()
            self._window = None
        self._close_requested = False
        self._paused = False