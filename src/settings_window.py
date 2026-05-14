"""
Settings Window
===============
Provides a simple settings panel accessible from the system tray
right-click menu. Exposes four behaviour toggles and a configurable
capture hotkey.

Changes take effect immediately on save. The hotkey listener is
rebuilt with the new key combination if the hotkey is changed.

Author: Harry Larkin
Date: March 2026
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional

from app_settings import AppSettings


class SettingsWindow:
    """
    Settings panel with four toggles and a hotkey entry field.

    Toggles:
        - Enable image preprocessing
        - Redact PII by default
        - Show live OCR visualisation
        - Auto-copy to clipboard after capture

    The on_save callback is called after settings are written so the
    main application can rebuild the hotkey listener if needed.

    Args:
        parent:   Parent Tk root window.
        settings: AppSettings instance to read from and write to.
        on_save:  Callback fired after settings are saved successfully.
    """

    def __init__(self, parent, settings: AppSettings, on_save: Optional[Callable] = None):
        self.parent   = parent
        self.settings = settings
        self.on_save  = on_save
        self.window   = None

        # BooleanVars for each toggle, created when the window is built
        self._pre_var  = None
        self._pii_var  = None
        self._viz_var  = None
        self._clip_var = None
        self._hk_var   = None

    def show(self):
        """
        Display the settings window. If already open, bring it to the front
        rather than creating a second instance.
        """
        if self.window and self.window.winfo_exists():
            self.window.lift()
            return

        self.window = tk.Toplevel(self.parent)
        self.window.title("Settings — Screenshot OCR Tool")
        self.window.resizable(False, False)

        # Centre on screen
        w, h = 360, 280
        x    = (self.window.winfo_screenwidth()  - w) // 2
        y    = (self.window.winfo_screenheight() - h) // 2
        self.window.geometry(f"{w}x{h}+{x}+{y}")
        self.window.lift()
        self.window.focus_force()

        self._build_ui()

    def _build_ui(self):
        """Build toggles, hotkey entry and Save/Cancel buttons."""
        pad = tk.Frame(self.window, padx=16, pady=12)
        pad.pack(fill='both', expand=True)

        # Initialise BooleanVars from current settings
        self._pre_var  = tk.BooleanVar(value=self.settings.preprocessing)
        self._pii_var  = tk.BooleanVar(value=self.settings.pii_redaction)
        self._viz_var  = tk.BooleanVar(value=self.settings.visualisation)
        self._clip_var = tk.BooleanVar(value=self.settings.auto_clipboard)

        # Render each toggle as a Checkbutton
        toggles = [
            ("Enable image preprocessing",           self._pre_var),
            ("Redact PII by default",                self._pii_var),
            ("Show live OCR visualisation",          self._viz_var),
            ("Auto-copy to clipboard after capture", self._clip_var),
        ]

        for label_text, var in toggles:
            tk.Checkbutton(
                pad,
                text=label_text,
                variable=var,
                font=("Segoe UI", 10),
                anchor='w'
            ).pack(fill='x', pady=3)

        ttk.Separator(pad, orient='horizontal').pack(fill='x', pady=8)

        # Hotkey entry — free text, parsed on save by _settings_to_pynput
        hk_frame = tk.Frame(pad)
        hk_frame.pack(fill='x')

        tk.Label(
            hk_frame,
            text="Capture hotkey:",
            font=("Segoe UI", 9)
        ).pack(side='left')

        self._hk_var = tk.StringVar(value=self.settings.hotkey_display)
        tk.Entry(
            hk_frame,
            textvariable=self._hk_var,
            font=("Consolas", 9),
            width=18
        ).pack(side='left', padx=8)

        # Save and Cancel buttons
        btn_frame = tk.Frame(pad)
        btn_frame.pack(fill='x', pady=(16, 0))

        ttk.Button(
            btn_frame,
            text="Save",
            command=self._save
        ).pack(side='right')

        ttk.Button(
            btn_frame,
            text="Cancel",
            command=self.window.destroy
        ).pack(side='right', padx=6)

    def _save(self):
        """
        Write all toggle and hotkey values back to AppSettings,
        fire the on_save callback, then close the window.
        """
        self.settings.preprocessing  = self._pre_var.get()
        self.settings.pii_redaction  = self._pii_var.get()
        self.settings.visualisation  = self._viz_var.get()
        self.settings.auto_clipboard = self._clip_var.get()
        self.settings.hotkey_display = self._hk_var.get()

        # Notify the main app so it can rebuild the hotkey listener
        if self.on_save:
            self.on_save()

        self.window.destroy()
        messagebox.showinfo("Settings", "Settings saved.")