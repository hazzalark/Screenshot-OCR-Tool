"""
Screenshot Module
==================
Provides full-screen and region-based screenshot capture, and a
fullscreen drag-and-drop region selector overlay.

Capture functions:
    capture_screenshot()         — captures the entire primary display
    capture_region_screenshot()  — captures a specific screen region

Region selection:
    RegionSelector               — fullscreen Tkinter overlay for
                                   drag-and-drop region selection with
                                   DPI-aware coordinate transformation

DPI scaling:
    On Windows displays with scaling above 100%, Tkinter reports logical
    pixel coordinates while PIL ImageGrab captures physical pixels. For
    example, on a 125% scaled 1920x1080 display, Tkinter reports 1536x864.
    RegionSelector detects this mismatch and applies a scale factor to all
    coordinates before capturing, producing pixel-perfect results across
    100%, 125% and 150% scaling configurations.

Author: Harry Larkin
Project: Screenshot OCR Tool (CI601)
Date: January 2026
"""

import os
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from PIL import ImageGrab

# ── Windows DPI awareness ──────────────────────────────────────────────────────
# Must be set before any screen size queries are made. Tells Windows to
# report physical pixel dimensions rather than scaled logical dimensions,
# which is required for the DPI scale factor calculation to work correctly.
if sys.platform == 'win32':
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


# ── Capture functions ──────────────────────────────────────────────────────────

def capture_screenshot(save_path: str = "screenshots") -> Optional[str]:
    """
    Capture the entire screen as a PNG and save it to disk.

    Uses PIL ImageGrab.grab() with no bounding box to capture the full
    primary display. The file is saved with a timestamp in the filename
    to avoid overwriting previous captures.

    Args:
        save_path: Directory to save the screenshot. Created automatically
                   if it does not exist.

    Returns:
        Absolute path to the saved PNG file, or None if the capture failed.
    """
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Created screenshots directory: {save_path}")

        print("Capturing full screen…")
        screenshot = ImageGrab.grab()

        # Timestamped filename prevents collisions between captures
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath  = os.path.join(save_path, f"screenshot_{timestamp}.png")

        screenshot.save(filepath, "PNG")
        print(f"Screenshot saved: {filepath}")
        print(f"  Resolution: {screenshot.size[0]}x{screenshot.size[1]}")
        return filepath

    except Exception as e:
        print(f"Full screen capture failed: {e}")
        return None


def capture_region_screenshot(
    bbox: tuple,
    save_path: str = "screenshots"
) -> Optional[str]:
    """
    Capture a specific rectangular region of the screen as a PNG.

    The bounding box coordinates must be in physical pixels (not logical
    pixels). RegionSelector applies the DPI scale factor before calling
    this function to ensure correct coordinates on scaled displays.

    Args:
        bbox:      Region as (left, top, right, bottom) in physical pixels.
        save_path: Directory to save the screenshot. Created automatically
                   if it does not exist.

    Returns:
        Absolute path to the saved PNG file, or None if the capture failed.
    """
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(f"Capturing region: {bbox}")
        screenshot = ImageGrab.grab(bbox=bbox)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath  = os.path.join(save_path, f"screenshot_region_{timestamp}.png")

        screenshot.save(filepath, "PNG")
        print(f"Region screenshot saved: {filepath}")
        print(f"  Resolution: {screenshot.size[0]}x{screenshot.size[1]}")
        return filepath

    except Exception as e:
        print(f"Region capture failed: {e}")
        return None


# ── Region selector ────────────────────────────────────────────────────────────

class RegionSelector:
    """
    Fullscreen semi-transparent Tkinter overlay for drag-and-drop region
    selection.

    Displays a crosshair cursor and instruction text. As the user drags,
    a red dashed rectangle is drawn showing the selected area with its
    dimensions in physical pixels. Releasing the mouse finalises the
    selection. ESC cancels.

    DPI scaling:
        Compares PIL and Tkinter reported resolutions to derive a scale
        factor, then multiplies all Tkinter coordinates by this factor
        before capturing. Validated across 100%, 125% and 150% scaling.

    ESC handling:
        overrideredirect(True) removes the window border and prevents
        the window manager from assigning keyboard focus automatically.
        ESC is bound on both root and canvas with focus_force() to ensure
        it always registers.
    """

    def __init__(self):
        self.root         = None
        self.canvas       = None
        self.start_x      = None
        self.start_y      = None
        self.current_rect = None   # Canvas item ID of the selection rectangle
        self.is_dragging  = False
        self.bbox         = None   # Set on successful selection, None if cancelled
        self.scale_factor = 1.0    # DPI scale factor

    # ── DPI scaling ────────────────────────────────────────────────────────────

    def get_scale_factor(self) -> float:
        """
        Calculate the DPI scale factor by comparing PIL and Tkinter sizes.

        PIL ImageGrab reports physical pixels (true display resolution).
        Tkinter reports logical pixels (resolution / scaling percentage).
        Dividing physical by logical gives the conversion factor.

        Example on a 125% scaled 1920x1080 display:
            PIL:     1920 x 1080  (physical)
            Tkinter: 1536 x 864   (logical)
            Factor:  1920/1536 = 1.25

        Returns:
            Float scale factor (1.0 on unscaled displays).
        """
        try:
            probe          = ImageGrab.grab()
            actual_w, actual_h = probe.size
            tk_w = self.root.winfo_screenwidth()
            tk_h = self.root.winfo_screenheight()
            scale_x = actual_w / tk_w
            scale_y = actual_h / tk_h
            print(f"DPI scale: PIL={actual_w}x{actual_h}, "
                  f"Tkinter={tk_w}x{tk_h}, "
                  f"factor={((scale_x + scale_y) / 2):.2f}")
            return (scale_x + scale_y) / 2
        except Exception:
            return 1.0

    # ── Mouse event handlers ───────────────────────────────────────────────────

    def on_mouse_down(self, event):
        """Record drag start position and clear any existing selection."""
        self.start_x     = event.x
        self.start_y     = event.y
        self.is_dragging = True
        if self.current_rect:
            self.canvas.delete(self.current_rect)

    def on_mouse_move(self, event):
        """
        Redraw the selection rectangle and update the dimension label.
        Dimensions are shown in physical pixels (after DPI scaling).
        """
        if not self.is_dragging:
            return
        if self.current_rect:
            self.canvas.delete(self.current_rect)

        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red', width=3, dash=(5, 5)
        )

        # Display physical pixel dimensions at the centre of the selection
        physical_w = int(abs(event.x - self.start_x) * self.scale_factor)
        physical_h = int(abs(event.y - self.start_y) * self.scale_factor)
        self.canvas.delete("dimension_text")
        self.canvas.create_text(
            (self.start_x + event.x) // 2,
            (self.start_y + event.y) // 2,
            text=f"{physical_w} x {physical_h}",
            fill="red",
            font=("Arial", 16, "bold"),
            tags="dimension_text"
        )

    def on_mouse_up(self, event):
        """
        Finalise the selection on mouse release.

        Converts Tkinter logical coordinates to physical pixels by applying
        the DPI scale factor. Rejects selections smaller than 10x10 pixels.
        """
        if not self.is_dragging:
            return
        self.is_dragging = False

        # Normalise so (left, top) is always the smaller corner
        left_tk   = min(self.start_x, event.x)
        top_tk    = min(self.start_y, event.y)
        right_tk  = max(self.start_x, event.x)
        bottom_tk = max(self.start_y, event.y)

        # Convert from logical to physical pixels
        left   = int(left_tk   * self.scale_factor)
        top    = int(top_tk    * self.scale_factor)
        right  = int(right_tk  * self.scale_factor)
        bottom = int(bottom_tk * self.scale_factor)

        if right - left > 10 and bottom - top > 10:
            self.bbox = (left, top, right, bottom)
            print(f"Region selected: ({left}, {top}) → ({right}, {bottom})  "
                  f"({right - left}x{bottom - top}px)")
            self.root.quit()
            self.root.destroy()
        else:
            print("Selection too small — please try again")
            if self.current_rect:
                self.canvas.delete(self.current_rect)
                self.canvas.delete("dimension_text")

    def on_key_press(self, event):
        """Cancel the selection if ESC is pressed."""
        if event.keysym == 'Escape':
            print("Region selection cancelled")
            self.bbox = None
            self.root.quit()
            self.root.destroy()

    # ── Main entry point ───────────────────────────────────────────────────────

    def select_region(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Open the fullscreen overlay and wait for the user to select a region.

        Returns:
            (left, top, right, bottom) in physical pixels, or None if cancelled.
        """
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)
        self.root.attributes('-topmost', True)
        self.root.configure(bg='black')
        self.root.overrideredirect(True)
        self.root.update_idletasks()

        # Calculate DPI scale now that the window exists
        self.scale_factor = self.get_scale_factor()

        self.canvas = tk.Canvas(
            self.root, bg='black',
            highlightthickness=0, cursor='crosshair', bd=0
        )
        self.canvas.pack(fill='both', expand=True)

        # Instruction text
        screen_width = self.root.winfo_screenwidth()
        self.canvas.create_text(
            screen_width // 2, 30,
            text="Click and drag to select region  •  ESC to cancel",
            fill="white", font=("Arial", 18, "bold")
        )

        self.canvas.bind('<Button-1>',        self.on_mouse_down)
        self.canvas.bind('<B1-Motion>',       self.on_mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)

        # Bind ESC on both root and canvas — overrideredirect prevents
        # automatic focus assignment so we must force it manually
        self.root.bind('<Key>',   self.on_key_press)
        self.canvas.bind('<Key>', self.on_key_press)
        self.root.focus_force()
        self.canvas.focus_set()

        self.root.mainloop()
        return self.bbox