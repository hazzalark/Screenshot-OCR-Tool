"""
Tesseract Manager
==================
Handles Tesseract OCR engine discovery, path configuration and
verification. Used by OCREngine via lazy initialisation — Tesseract
is not loaded until the first OCR call, keeping application startup fast.

Path detection searches in order:
    1. Bundled executable (sys._MEIPASS when running as a PyInstaller EXE)
    2. System PATH (via shutil.which)
    3. Portable installation in the project directory (./tesseract/)
    4. Common platform-specific installation directories

Author: Harry Larkin
Date: January 2026
"""

import os
import sys
import shutil
import platform
from pathlib import Path
from typing import Optional

import pytesseract


class TesseractManager:
    """
    Manages Tesseract OCR engine discovery, path configuration and
    runtime verification.

    Designed to be used by OCREngine via a lazy initialisation pattern.
    The initialise() method is a no-op on subsequent calls so it can be
    called safely before every OCR operation.
    """

    def __init__(self, custom_path: Optional[str] = None):
        """
        Args:
            custom_path: Optional explicit path to the Tesseract executable.
                         If provided, skips auto-detection entirely.
        """
        self._custom_path = custom_path
        self._ready       = False   # Lazy init flag — True once Tesseract is verified

    # ── Public API ─────────────────────────────────────────────────────────────

    def initialise(self):
        """
        Initialise Tesseract on first call. Subsequent calls return immediately.

        Resolves the executable path, sets pytesseract.tesseract_cmd, and
        sets TESSDATA_PREFIX when running as a compiled EXE so Tesseract
        can find its language data files.
        """
        if self._ready:
            return

        # Resolve path: custom > auto-detect
        path = self._custom_path or self._find_tesseract()
        if path:
            pytesseract.pytesseract.tesseract_cmd = path

        # When running as a PyInstaller EXE, set TESSDATA_PREFIX so Tesseract
        # can locate the bundled tessdata directory inside sys._MEIPASS
        if getattr(sys, 'frozen', False):
            tessdata_path = os.path.join(sys._MEIPASS, 'tesseract', 'tessdata')
            os.environ['TESSDATA_PREFIX'] = tessdata_path

        self._verify()
        self._ready = True

    @property
    def is_ready(self) -> bool:
        """True if Tesseract has been successfully initialised."""
        return self._ready

    # ── Path detection ─────────────────────────────────────────────────────────

    def _find_tesseract(self) -> Optional[str]:
        """
        Auto-detect the Tesseract executable by searching common locations.

        Returns:
            Absolute path string to the Tesseract executable, or None if
            not found in any of the searched locations.
        """
        # 1. Bundled executable inside a PyInstaller compiled EXE
        if getattr(sys, 'frozen', False):
            bundled = os.path.join(sys._MEIPASS, 'tesseract', 'tesseract.exe')
            if os.path.exists(bundled):
                print(f"Tesseract found (bundled): {bundled}")
                return bundled

        # 2. System PATH — works on all platforms if Tesseract is installed globally
        system_path = shutil.which('tesseract')
        if system_path:
            print(f"Tesseract found (system PATH): {system_path}")
            return system_path

        # 3. Portable installation bundled alongside the project source
        project_root = Path(__file__).parent.parent  # src/ocr/ → src/
        portable_candidates = [
            project_root / "tesseract" / "tesseract.exe",
            project_root / "tesseract" / "tesseract",
            project_root.parent / "tesseract" / "tesseract.exe",
            project_root.parent / "tesseract" / "tesseract",
        ]
        for candidate in portable_candidates:
            if candidate.exists():
                print(f"Tesseract found (portable): {candidate}")
                return str(candidate)

        # 4. Common platform-specific installation directories
        system = platform.system()

        if system == 'Windows':
            windows_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Tesseract-OCR\tesseract.exe",
            ]
            for path in windows_paths:
                if os.path.exists(path):
                    print(f"Tesseract found (Windows install): {path}")
                    return path

        elif system in ('Linux', 'Darwin'):
            unix_paths = [
                '/usr/bin/tesseract',
                '/usr/local/bin/tesseract',
                '/opt/homebrew/bin/tesseract',
            ]
            for path in unix_paths:
                if os.path.exists(path):
                    print(f"Tesseract found (Unix install): {path}")
                    return path

        print("Tesseract not found in any known location.")
        return None

    # ── Verification ───────────────────────────────────────────────────────────

    def _verify(self):
        """
        Verify Tesseract is accessible by requesting its version string.

        Raises:
            RuntimeError: If Tesseract cannot be called, with installation
                          instructions for Windows, Linux and Mac.
        """
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract OCR ready — version {version}")
        except Exception as e:
            raise RuntimeError(
                "Tesseract OCR is not accessible.\n"
                "Please install Tesseract:\n"
                "  Windows : https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  Linux   : sudo apt-get install tesseract-ocr\n"
                "  Mac     : brew install tesseract\n"
                f"Error: {e}"
            )