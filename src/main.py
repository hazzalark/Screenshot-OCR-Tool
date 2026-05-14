"""
Screenshot OCR Tool - Console Entry Point
==========================================
This file is the console/development entry point for the Screenshot OCR Tool.

For the full GUI application (system tray, hotkey, live visualisation),
run gui.py instead:
    python src/gui.py

This file remains useful for:
- Development and debugging without the GUI
- Testing individual modules from the command line
- Running OCR from scripts or other tools

Author: Harry Larkin
Project: Screenshot OCR Tool (CI601)
Date: January 2026
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

if __name__ == "__main__":
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from screenshot import RegionSelector, capture_region_screenshot
from ocr_engine import OCREngine
from export import OCRExporter
from categorization import TextCategoriser, CATEGORIES
from pii_detection import PIIDetector

try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False
    print("pyperclip not available - clipboard features disabled")


class ScreenshotOCRTool:
    """
    Console version of the Screenshot OCR Tool.
    For the GUI version see gui.py.
    """

    def __init__(self, preprocessing: bool = True):
        """
        Initialise the Screenshot OCR Tool.

        Tesseract is not loaded until the first screenshot is processed
        (lazy initialisation).

        Args:
            preprocessing: Enable OCR image preprocessing by default.
        """
        self.ocr_engine   = OCREngine()
        self.ocr_engine.set_preprocessing(preprocessing)
        self.exporter     = OCRExporter()
        self.categoriser  = TextCategoriser()
        self.pii_detector = PIIDetector(redact_by_default=True)

        self.last_result          = None
        self.last_pii_result      = None
        self.last_screenshot_path = None

    # ── Core workflow ──────────────────────────────────────────────────────────

    def capture_and_extract(
        self,
        save_screenshot: bool = True,
        screenshot_dir: str = "screenshots"
    ) -> Optional[Dict[str, Any]]:
        """
        Launch the region selector, capture the selected area, run OCR,
        categorise the text, and scan for PII.

        Tesseract initialises here on first call, not at app startup.

        Args:
            save_screenshot: Whether to save the captured PNG to disk.
            screenshot_dir:  Directory for saved screenshots.

        Returns:
            Result dict or None if cancelled.
        """
        print("\n" + "=" * 70)
        print("SCREENSHOT OCR TOOL")
        print("=" * 70)
        print("\nStarting region selection...")
        print("  - Click and drag to select region")
        print("  - Press ESC to cancel")

        selector = RegionSelector()
        bbox     = selector.select_region()

        if not bbox:
            print("\nCancelled.")
            return None

        print(f"\nRegion selected: {bbox}")

        if save_screenshot:
            screenshot_path           = capture_region_screenshot(bbox, screenshot_dir)
            self.last_screenshot_path = screenshot_path
        else:
            from PIL import ImageGrab
            image           = ImageGrab.grab(bbox=bbox)
            screenshot_path = None

        if not screenshot_path:
            print("Screenshot capture failed.")
            return None

        print(f"Screenshot saved: {screenshot_path}")

        # OCR — Tesseract initialises here on first run
        print("\nPerforming OCR...")
        result         = self.ocr_engine.extract_text_from_file(screenshot_path)
        extracted_text = result.get('text', '')

        # Categorisation
        if extracted_text.strip():
            print("\nCategorising text...")
            category_result    = self.categoriser.categorise(extracted_text)
            result['category'] = category_result
            print(
                f"Category: {category_result['category_label']} "
                f"({category_result['confidence_pct']}, {category_result['method']})"
            )
        else:
            result['category'] = {
                'category':         'documentation',
                'category_label':   CATEGORIES['documentation'],
                'confidence':       0.0,
                'confidence_pct':   '0%',
                'confidence_level': 'Low',
                'method':           'none',
                'all_scores':       {},
            }

        # PII Detection
        print("\nScanning for PII...")
        pii_result           = self.pii_detector.process(extracted_text)
        self.last_pii_result = pii_result
        result['pii']        = pii_result
        self.pii_detector.print_summary(pii_result)

        result['screenshot_path'] = screenshot_path
        result['bbox']            = bbox
        result['timestamp']       = datetime.now().isoformat()

        self.last_result = result
        self._display_results(result)
        return result

    # ── Display ───────────────────────────────────────────────────────────────

    def _display_results(self, result: Dict[str, Any]):
        """Print a formatted summary of OCR, category and PII results."""
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        cat = result.get('category', {})
        pii = result.get('pii')

        print(f"\n  Processing time : {result['processing_time']:.2f}s")
        print(f"  OCR confidence  : {result['confidence']:.1f}%")
        print(f"  Characters      : {result['char_count']}")
        print(f"  Words           : {result['word_count']}")
        print(f"  Preprocessing   : {'Enabled' if result['preprocessing_used'] else 'Disabled'}")
        print(f"  Category        : {cat.get('category_label', 'Unknown')} "
              f"({cat.get('confidence_pct', '?')} via {cat.get('method', '?')})")
        if pii and pii.has_pii:
            print(f"  PII             : {pii.summary}")
        else:
            print(f"  PII             : None detected")

        print("\n" + "-" * 70)
        print("EXTRACTED TEXT")
        print("-" * 70)
        for i, line in enumerate(result['text'].split('\n'), 1):
            if line.strip():
                print(f"{i:3d} | {line}")
        print("-" * 70)

    # ── PII helpers ───────────────────────────────────────────────────────────

    def _get_pii_choice(self) -> str:
        """Ask the user whether to redact PII before export. Returns 'redact' or 'keep'."""
        pii = self.last_pii_result
        if not pii or not pii.has_pii:
            return 'keep'
        print(f"\n  {pii.summary}")
        print("  Redact PII before export? (y/n, default y): ", end='')
        choice = input().strip().lower()
        return 'keep' if choice == 'n' else 'redact'

    def _get_export_text(self, pii_choice: str) -> str:
        """Return the correct text for export based on PII choice."""
        if not self.last_pii_result:
            return self.last_result['text']
        return self.pii_detector.get_export_text(self.last_pii_result, pii_choice)

    # ── Export helpers ────────────────────────────────────────────────────────

    def export_to_clipboard(self) -> bool:
        if not self.last_result:
            print("No results to export.")
            return False
        return self.exporter.export_to_clipboard(
            self._get_export_text(self._get_pii_choice())
        )

    def export_to_txt(self, filepath=None) -> Optional[str]:
        if not self.last_result:
            print("No results to export.")
            return None
        return self.exporter.export_to_txt(
            self._get_export_text(self._get_pii_choice()),
            filepath, include_metadata=True, metadata=self.last_result
        )

    def export_to_docx(self, filepath=None) -> Optional[str]:
        if not self.last_result:
            print("No results to export.")
            return None
        return self.exporter.export_to_docx(
            self._get_export_text(self._get_pii_choice()),
            filepath, include_metadata=True, metadata=self.last_result
        )

    def export_to_json(self, filepath=None) -> Optional[str]:
        if not self.last_result:
            print("No results to export.")
            return None
        return self.exporter.export_to_json(
            self._get_export_text(self._get_pii_choice()),
            filepath, metadata=self.last_result
        )

    def export_all_formats(self) -> Dict[str, Optional[str]]:
        if not self.last_result:
            print("No results to export.")
            return {}
        pii_choice = self._get_pii_choice()
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.exporter.export_all_formats(
            self._get_export_text(pii_choice),
            metadata=self.last_result,
            base_filename=f"ocr_export_{timestamp}"
        )

    # ── Interactive menu ──────────────────────────────────────────────────────

    def interactive_menu(self):
        """Display post-capture menu for export and further actions."""
        if not self.last_result:
            print("No results available.")
            return

        cat = self.last_result.get('category', {})
        pii = self.last_pii_result

        print("\n" + "=" * 70)
        print("EXPORT OPTIONS")
        print("=" * 70)
        print(f"\n  Category : {cat.get('category_label', 'Unknown')} ({cat.get('confidence_pct', '?')})")
        if pii and pii.has_pii:
            print(f"  PII      : {pii.summary}")
        else:
            print(f"  PII      : None detected")
        print()
        print("  1. Copy to clipboard")
        print("  2. Save as .txt")
        print("  3. Save as .docx")
        print("  4. Save as .json (with metadata)")
        print("  5. Export all formats")
        print("  6. New capture")
        print("  7. Exit")

        choice = input("\nEnter choice (1-7): ").strip()

        if choice == '1':
            self.export_to_clipboard()
            self.interactive_menu()
        elif choice == '2':
            self.export_to_txt()
            self.interactive_menu()
        elif choice == '3':
            self.export_to_docx()
            self.interactive_menu()
        elif choice == '4':
            self.export_to_json()
            self.interactive_menu()
        elif choice == '5':
            self.export_all_formats()
            self.interactive_menu()
        elif choice == '6':
            self.capture_and_extract()
            self.interactive_menu()
        elif choice == '7':
            print("\nGoodbye!")
        else:
            print("Invalid choice.")
            self.interactive_menu()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SCREENSHOT OCR TOOL  (Console Mode)")
    print("For the GUI version run: python src/gui.py")
    print("=" * 70)
    tool   = ScreenshotOCRTool(preprocessing=True)
    result = tool.capture_and_extract()
    if result:
        tool.interactive_menu()
    else:
        print("\nNo screenshot captured.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

