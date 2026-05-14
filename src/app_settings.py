"""
Application Settings
====================
Holds all user configurable settings for the Screenshot OCR Tool.
Settings are passed between the main application, windows and modules
to keep behaviour consistent across the session.

Author: Harry Larkin
Project: Screenshot OCR Tool (CI601)
Date: March 2026
"""


class AppSettings:
    """
    Stores all user configurable application settings with sensible defaults.

    Settings are modified via the SettingsWindow and take effect immediately
    on save. The hotkey listener is rebuilt when the hotkey changes.
    """

    def __init__(self):
        # Hotkey displayed in the tray menu and settings window
        self.hotkey_display = "Ctrl+Shift+S"

        # Whether to apply the OCR image preprocessing pipeline
        self.preprocessing  = True

        # Whether to redact PII by default before export
        self.pii_redaction  = True

        # Whether to show the live OCR bounding box visualiser
        self.visualisation  = True

        # Whether to automatically copy extracted text to clipboard after capture
        self.auto_clipboard = False