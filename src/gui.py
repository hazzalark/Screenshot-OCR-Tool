"""
Screenshot OCR Tool - Main Application
=======================================
System tray application that orchestrates the full capture pipeline:
region selection, OCR, categorisation, PII detection and export.

On launch the application minimises directly to the system tray with
no visible window. Captures are triggered via the configurable hotkey
(default Ctrl+Shift+S) or the tray right-click menu.

Author: Harry Larkin
Project: Screenshot OCR Tool (CI601)
Date: January 2026
"""

import os
import sys
import threading
import traceback
import tkinter as tk
from pathlib import Path
from datetime import datetime
from typing import Dict

# ── Path setup ─────────────────────────────────────────────────────────────────
_base = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).parent
if str(_base) not in sys.path:
    sys.path.insert(0, str(_base))

# ── Silent error logging (EXE only) ───────────────────────────────────────────

def _get_app_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).parent


_log_path = _get_app_dir() / "error.log"

if getattr(sys, 'frozen', False):
    class _FileLogger:
        def __init__(self, path: Path):
            self._path = path
        def write(self, msg: str):
            if msg.strip():
                try:
                    with open(self._path, 'a', encoding='utf-8') as f:
                        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
                except Exception:
                    pass
        def flush(self):
            pass
    sys.stdout = _FileLogger(_log_path)
    sys.stderr = _FileLogger(_log_path)

# ── Local imports ──────────────────────────────────────────────────────────────
from screenshot import RegionSelector, capture_region_screenshot
from ocr_engine import OCREngine
from export import OCRExporter
from categorization import TextCategoriser, CATEGORIES
from pii_detection import PIIDetector
from app_settings import AppSettings
from visualiser import LiveOCRVisualiser
from results_window import ResultsWindow
from settings_window import SettingsWindow

try:
    import pystray
    from pystray import MenuItem as TrayItem
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False

try:
    from pynput import keyboard
    HOTKEY_AVAILABLE = True
except ImportError:
    HOTKEY_AVAILABLE = False

from PIL import Image, ImageDraw


class ScreenshotOCRApp:
    """
    Main application class. Owns the hidden Tk root window, system tray
    icon and global hotkey listener.
    """

    def __init__(self):
        print("Initialising app...")
        self.settings     = AppSettings()
        self.ocr_engine   = OCREngine()
        self.exporter     = OCRExporter()
        self.categoriser  = TextCategoriser()
        self.pii_detector = PIIDetector(redact_by_default=True)
        self._capturing       = False
        self._tray            = None
        self._hotkey_listener = None
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Screenshot OCR Tool")
        print("App initialised successfully")

    def _make_tray_icon(self) -> Image.Image:
        size = 64
        img  = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        d    = ImageDraw.Draw(img)
        d.ellipse([2, 2, size - 2, size - 2], fill="#0078d4")
        d.rectangle([18, 26, 46, 38], fill="white")
        return img

    def _build_tray(self):
        if not TRAY_AVAILABLE:
            print("pystray not available — tray disabled")
            return

        print("pystray available, building menu...")
        menu = pystray.Menu(
            TrayItem(
                f"Capture  ({self.settings.hotkey_display})",
                lambda: self._trigger_capture()
            ),
            pystray.Menu.SEPARATOR,
            TrayItem("Settings", lambda: self.root.after(0, self._open_settings)),
            pystray.Menu.SEPARATOR,
            TrayItem("Exit", lambda: self._quit()),
        )
        self._tray = pystray.Icon(
            "ScreenshotOCR",
            self._make_tray_icon(),
            "Screenshot OCR Tool",
            menu
        )
        print(f"Tray icon created: {self._tray}")
        threading.Thread(target=self._tray.run, daemon=True).start()
        print("Tray thread started")

    def _settings_to_pynput(self, display: str) -> str:
        modifiers = {'ctrl', 'shift', 'alt', 'cmd', 'win'}
        parts     = [p.strip().lower() for p in display.split('+')]
        mapped    = [f"<{p}>" if p in modifiers else p for p in parts]
        return '+'.join(mapped)

    def _build_hotkey(self):
        if not HOTKEY_AVAILABLE:
            return
        try:
            if self._hotkey_listener:
                self._hotkey_listener.stop()
                self._hotkey_listener = None
            hotkey_str = self._settings_to_pynput(self.settings.hotkey_display)
            self._hotkey_listener = keyboard.GlobalHotKeys(
                {hotkey_str: self._trigger_capture}
            )
            self._hotkey_listener.start()
            print(f"Hotkey registered: {self.settings.hotkey_display} → {hotkey_str}")
        except Exception as e:
            print(f"Hotkey setup failed: {e}\n{traceback.format_exc()}")

    def _trigger_capture(self):
        if not self._capturing:
            self.root.after(0, self._do_capture)

    def _do_capture(self):
        if self._capturing:
            return
        self._capturing = True
        try:
            selector = RegionSelector()
            bbox     = selector.select_region()
            if not bbox:
                self._capturing = False
                return
            screenshot_path = capture_region_screenshot(bbox, "screenshots")
            if not screenshot_path:
                self._capturing = False
                return
            self.ocr_engine.set_preprocessing(self.settings.preprocessing)
            if self.settings.visualisation:
                viz = LiveOCRVisualiser(
                    screenshot_path,
                    on_complete=lambda r: self.root.after(
                        0, lambda: self._on_ocr_complete(r, screenshot_path)
                    ),
                    ocr_engine=self.ocr_engine
                )
                viz.start()
            else:
                result = self.ocr_engine.extract_text_from_file(screenshot_path)
                self._on_ocr_complete(result, screenshot_path)
        except Exception as e:
            print(f"Capture error: {e}\n{traceback.format_exc()}")
            self._capturing = False

    def _on_ocr_complete(self, ocr_result: Dict, screenshot_path: str):
        try:
            text = ocr_result.get('text', '')
            category_result = (
                self.categoriser.categorise(text)
                if text.strip()
                else {
                    'category':         'documentation',
                    'category_label':   CATEGORIES['documentation'],
                    'confidence':       0.0,
                    'confidence_pct':   '0%',
                    'confidence_level': 'Low',
                    'method':           'none',
                    'all_scores':       {}
                }
            )
            pii_result = self.pii_detector.process(
                text, redact=self.settings.pii_redaction
            )
            if self.settings.auto_clipboard:
                self.exporter.export_to_clipboard(
                    self.pii_detector.get_export_text(
                        pii_result,
                        'redact' if self.settings.pii_redaction else 'keep'
                    )
                )
            ocr_result['screenshot_path'] = screenshot_path
            ocr_result['timestamp']       = datetime.now().isoformat()
            ocr_result['category']        = category_result
            ResultsWindow(
                self.root, ocr_result, pii_result,
                category_result, self.exporter,
                self.pii_detector, self.settings
            ).show()
        except Exception as e:
            print(f"Post-OCR error: {e}\n{traceback.format_exc()}")
        finally:
            self._capturing = False

    def _open_settings(self):
        SettingsWindow(
            self.root, self.settings,
            on_save=self._on_settings_saved
        ).show()

    def _on_settings_saved(self):
        self._build_hotkey()
        self.pii_detector.redact_by_default = self.settings.pii_redaction

    def _quit(self):
        if self._tray:
            self._tray.stop()
        if self._hotkey_listener:
            self._hotkey_listener.stop()
        self.root.quit()

    def run(self):
        print("Building tray...")
        self._build_tray()
        print("Tray built, building hotkey...")
        self._build_hotkey()
        print("Hotkey built, entering mainloop...")
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self._quit()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    app = ScreenshotOCRApp()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            with open(_log_path, 'a') as f:
                f.write(f"\n[FATAL {datetime.now()}]\n{traceback.format_exc()}\n")
        except Exception:
            pass