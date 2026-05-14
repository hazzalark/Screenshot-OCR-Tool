"""
OCR Engine
===========
Provides text extraction from images and files using Tesseract OCR.
Integrates TesseractManager for lazy path initialisation and
ImagePreprocessor for accuracy improvement.

Tesseract configuration:
    PSM 3  — Fully automatic page segmentation (handles mixed layouts)
    OEM 1  — LSTM neural network only (best accuracy on clean text)

Lazy initialisation:
    Tesseract is not loaded at application startup. The first call to
    extract_text_from_image or extract_text_from_file triggers
    TesseractManager.initialise(), which detects the executable path
    and verifies the installation. Subsequent calls return immediately.

Author: Harry Larkin
Project: Screenshot OCR Tool (CI601)
Date: January 2026
"""

import os
import time
from typing import Optional, Dict, Any
from PIL import Image

from tesseract_manager import TesseractManager
from preprocessor import ImagePreprocessor
import pytesseract


class OCREngine:
    """
    Extracts text from images using Tesseract OCR with an optional
    preprocessing pipeline.

    Usage:
        engine = OCREngine()
        result = engine.extract_text_from_file('screenshot.png')
        print(result['text'], result['confidence'])

    The engine is safe to instantiate at application startup — Tesseract
    is not loaded until the first extraction call.
    """

    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Args:
            tesseract_path: Optional explicit path to the Tesseract executable.
                            If None, TesseractManager auto-detects on first use.
        """
        # Manages Tesseract path detection and verification
        self._tesseract    = TesseractManager(custom_path=tesseract_path)

        # Manages the image preprocessing pipeline
        self._preprocessor = ImagePreprocessor()

    # ── Preprocessing control ──────────────────────────────────────────────────

    def set_preprocessing(self, enabled: bool):
        """
        Enable or disable the entire preprocessing pipeline.

        When disabled, images are passed directly to Tesseract without
        any transformation. Useful for testing or high-quality inputs.

        Args:
            enabled: True to enable preprocessing, False to disable.
        """
        self._preprocessor.enabled = enabled
        print(f"OCR preprocessing {'enabled' if enabled else 'disabled'}")

    def configure_preprocessing(self, **kwargs):
        """
        Enable or disable individual preprocessing steps.

        Args:
            grayscale (bool):            Grayscale conversion
            noise_reduction (bool):      Median noise filter
            binarization (bool):         Otsu adaptive binarisation
            contrast_enhancement (bool): Contrast amplification
            deskew (bool):               Rotation correction (expensive)
        """
        self._preprocessor.configure(**kwargs)

    # ── Internal Tesseract init proxy ──────────────────────────────────────────

    def _ensure_tesseract(self):
        """
        Proxy to TesseractManager.initialise() used by the live visualiser.
        The visualiser calls this directly before making its own pytesseract
        calls to ensure the path is configured consistently.
        """
        self._tesseract.initialise()

    # ── Extraction ─────────────────────────────────────────────────────────────

    def extract_text_from_image(
        self,
        image:  Image.Image,
        lang:   str = 'eng',
        config: str = '--psm 3 --oem 1'
    ) -> Dict[str, Any]:
        """
        Extract text from a PIL Image using Tesseract OCR.

        Initialises Tesseract on the first call (lazy init). Applies the
        preprocessing pipeline if enabled, then runs Tesseract to extract
        text and per-word confidence scores.

        PSM modes:
            3  Fully automatic page segmentation (default)
            6  Single uniform block of text
            11 Sparse text

        OEM modes:
            1  LSTM neural network only — best accuracy (default)
            3  Legacy + LSTM combined

        Args:
            image:  PIL Image to extract text from.
            lang:   Tesseract language code (default: 'eng').
            config: Tesseract configuration flags.

        Returns:
            dict containing:
                text             — extracted text string
                confidence       — mean per-word confidence (0–100)
                processing_time  — elapsed time in seconds
                char_count       — character count of extracted text
                word_count       — word count of extracted text
                preprocessing_used — whether preprocessing was applied
        """
        self._tesseract.initialise()

        start = time.time()

        # Apply preprocessing pipeline if enabled
        processed = (
            self._preprocessor.process(image)
            if self._preprocessor.enabled
            else image
        )

        # Extract full text string
        text = pytesseract.image_to_string(processed, lang=lang, config=config)

        # Extract word-level data for per-word confidence scores
        # Tesseract returns -1 confidence for non-word elements (separators etc.)
        data = pytesseract.image_to_data(
            processed,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT
        )

        # Calculate mean confidence across all recognised words only
        confidences    = [int(c) for c in data['conf'] if int(c) != -1]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        elapsed        = time.time() - start

        print(
            f"OCR complete — {len(text.split())} words, "
            f"{avg_confidence:.1f}% confidence, "
            f"{elapsed:.2f}s"
        )

        return {
            'text':               text,
            'confidence':         avg_confidence,
            'processing_time':    elapsed,
            'char_count':         len(text.strip()),
            'word_count':         len(text.split()),
            'preprocessing_used': self._preprocessor.enabled,
        }

    def extract_text_from_file(
        self,
        filepath: str,
        lang:     str = 'eng',
        config:   str = '--psm 3 --oem 1'
    ) -> Dict[str, Any]:
        """
        Load an image file and extract text using Tesseract OCR.

        Convenience wrapper around extract_text_from_image that handles
        file loading and error reporting.

        Args:
            filepath: Path to the image file.
            lang:     Tesseract language code (default: 'eng').
            config:   Tesseract configuration flags.

        Returns:
            dict with the same structure as extract_text_from_image.

        Raises:
            FileNotFoundError: If the image file does not exist.
            RuntimeError:      If the file cannot be opened or OCR fails.
        """
        self._tesseract.initialise()

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found: {filepath}")

        try:
            image = Image.open(filepath)
            return self.extract_text_from_image(image, lang, config)
        except Exception as e:
            raise RuntimeError(f"Failed to process image file: {e}")

    def get_available_languages(self) -> list:
        """
        Return a list of installed Tesseract language codes.

        Returns:
            List of language code strings (e.g. ['eng', 'fra']).
            Falls back to ['eng'] if the list cannot be retrieved.
        """
        self._tesseract.initialise()
        try:
            return pytesseract.get_languages()
        except Exception:
            return ['eng']