"""
Results Window
==============
Displays the full output of the OCR capture pipeline after the live
visualiser closes. Shows extracted text, category, PII status and
provides export options.

Author: Harry Larkin
Date: March 2026
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from typing import Dict

from export import OCRExporter
from pii_detection import PIIDetector
from app_settings import AppSettings


class ResultsWindow:
    """
    Post-capture results window showing OCR output with category,
    PII status and export controls.

    Layout:
        - Info bar: category, word count, OCR confidence, processing time
        - PII notice: yellow warning bar if PII found, green bar if clean
        - Extracted text: read-only scrollable text box
        - Export row: Copy to Clipboard, Save As, Close (always visible)

    The export row is packed before the text area so it remains visible
    at any window height without needing to expand the window.

    Args:
        parent:          Parent Tk root window.
        result:          OCR result dict from OCREngine or LiveOCRVisualiser.
        pii_result:      PIIResult from PIIDetector.process().
        category_result: Category dict from TextCategoriser.categorise().
        exporter:        OCRExporter instance for file exports.
        pii_detector:    PIIDetector instance for redaction.
        settings:        AppSettings instance for default redaction preference.
    """

    def __init__(self, parent, result: Dict, pii_result,
                 category_result: Dict, exporter: OCRExporter,
                 pii_detector: PIIDetector, settings: AppSettings):
        self.parent          = parent
        self.result          = result
        self.pii_result      = pii_result
        self.category_result = category_result
        self.exporter        = exporter
        self.pii_detector    = pii_detector
        self.settings        = settings

        # BooleanVar controlling the redact checkbox (only created if PII found)
        self._redact_var = None

        self.window = None

    def show(self):
        """Create and display the results window, centred on screen."""
        self.window = tk.Toplevel(self.parent)
        self.window.title("OCR Results")
        self.window.resizable(True, True)
        self.window.minsize(500, 520)

        # Centre window on screen
        self.window.update_idletasks()
        w, h = 580, 620
        x    = (self.window.winfo_screenwidth()  - w) // 2
        y    = (self.window.winfo_screenheight() - h) // 2
        self.window.geometry(f"{w}x{h}+{x}+{y}")
        self.window.lift()
        self.window.focus_force()

        self._build_ui()

    def _build_ui(self):
        """Build all UI elements in the correct pack order."""
        w   = self.window
        cat = self.category_result
        pii = self.pii_result

        # ── Info bar ──────────────────────────────────────────────────────────
        # Shows category, word count, OCR confidence and processing time
        info = tk.Frame(w, bg='#f0f0f0', pady=6, padx=10)
        info.pack(fill='x')

        tk.Label(
            info,
            text=f"Category: {cat.get('category_label', '?')}  "
                 f"({cat.get('confidence_pct', '?')} confidence)",
            bg='#f0f0f0',
            font=("Segoe UI", 9)
        ).pack(side='left')

        tk.Label(
            info,
            text=f"{self.result['word_count']} words  ·  "
                 f"{self.result['confidence']:.0f}% OCR confidence  ·  "
                 f"{self.result['processing_time']:.2f}s",
            bg='#f0f0f0',
            font=("Segoe UI", 9)
        ).pack(side='right')

        # ── PII notice ────────────────────────────────────────────────────────
        # Yellow warning bar with redact toggle if PII was detected,
        # green confirmation bar if the text is clean
        if pii and pii.has_pii:
            pii_frame = tk.Frame(w, bg='#fff3cd', pady=4, padx=10)
            pii_frame.pack(fill='x')

            tk.Label(
                pii_frame,
                text=f"⚠  {pii.summary}",
                bg='#fff3cd',
                fg='#856404',
                font=("Segoe UI", 9)
            ).pack(side='left')

            # Redact checkbox defaults to the setting in AppSettings
            self._redact_var = tk.BooleanVar(value=self.settings.pii_redaction)
            tk.Checkbutton(
                pii_frame,
                text="Redact PII before export",
                variable=self._redact_var,
                bg='#fff3cd',
                font=("Segoe UI", 9)
            ).pack(side='right')
        else:
            notice = tk.Frame(w, bg='#d4edda', pady=4, padx=10)
            notice.pack(fill='x')
            tk.Label(
                notice,
                text="✓  No PII detected",
                bg='#d4edda',
                fg='#155724',
                font=("Segoe UI", 9)
            ).pack(side='left')

        ttk.Separator(w, orient='horizontal').pack(fill='x', pady=2)

        # ── Export row ────────────────────────────────────────────────────────
        # Packed with side='bottom' BEFORE the text area so it is always
        # visible regardless of window height. The text area fills remaining space.
        export_frame = tk.Frame(w, pady=8, padx=10)
        export_frame.pack(fill='x', side='bottom')

        ttk.Separator(w, orient='horizontal').pack(fill='x', side='bottom')

        ttk.Button(
            export_frame,
            text="Copy to Clipboard",
            command=self._export_clipboard
        ).pack(side='left', padx=(0, 6))

        ttk.Button(
            export_frame,
            text="Save As…",
            command=self._export_save_as
        ).pack(side='left', padx=3)

        ttk.Button(
            export_frame,
            text="Close",
            command=self.window.destroy
        ).pack(side='right')

        # ── Text area ─────────────────────────────────────────────────────────
        # Packed last so it expands to fill all remaining vertical space.
        # State is set to 'disabled' after inserting text to prevent editing.
        tk.Label(
            w,
            text="Extracted Text:",
            font=("Segoe UI", 9),
            anchor='w',
            padx=10
        ).pack(fill='x')

        text_frame = tk.Frame(w)
        text_frame.pack(fill='both', expand=True, padx=10, pady=(2, 4))

        self._text_box = tk.Text(
            text_frame,
            font=("Consolas", 10),
            relief='sunken',
            bd=1,
            wrap='word',
            padx=6,
            pady=6,
            state='normal',
        )

        scrollbar = ttk.Scrollbar(text_frame, command=self._text_box.yview)
        self._text_box.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self._text_box.pack(fill='both', expand=True)

        # Insert text then disable to make read-only
        self._text_box.insert('1.0', self.result['text'])
        self._text_box.configure(state='disabled')

    # ── Export helpers ─────────────────────────────────────────────────────────

    def _get_export_text(self) -> str:
        """
        Return the appropriate text for export based on the redact checkbox.
        If no PII was found, returns the original text unchanged.
        """
        redact = (
            self._redact_var.get()
            if self._redact_var is not None
            else self.settings.pii_redaction
        )
        return self.pii_detector.get_export_text(
            self.pii_result,
            'redact' if redact else 'keep'
        )

    def _export_clipboard(self):
        """Copy the export text to the system clipboard via pyperclip."""
        self.exporter.export_to_clipboard(self._get_export_text())
        messagebox.showinfo("Copied", "Text copied to clipboard.", parent=self.window)

    def _export_save_as(self):
        """
        Open a native Save As dialog allowing the user to choose filename,
        location and format (TXT, DOCX or JSON). The file extension determines
        which exporter is called.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        filepath = filedialog.asksaveasfilename(
            parent=self.window,
            title="Save extracted text",
            initialfile=f"ocr_export_{ts}",
            defaultextension=".txt",
            filetypes=[
                ("Text file",     "*.txt"),
                ("Word document", "*.docx"),
                ("JSON file",     "*.json"),
                ("All files",     "*.*"),
            ]
        )

        # User cancelled the dialog
        if not filepath:
            return

        ext  = filepath.rsplit('.', 1)[-1].lower() if '.' in filepath else 'txt'
        text = self._get_export_text()
        path = None

        if ext == 'docx':
            path = self.exporter.export_to_docx(
                text, filepath,
                include_metadata=True,
                metadata=self.result
            )
        elif ext == 'json':
            path = self.exporter.export_to_json(
                text, filepath,
                metadata=self.result
            )
        else:
            # Default to plain text for .txt or any unrecognised extension
            path = self.exporter.export_to_txt(
                text, filepath,
                include_metadata=True,
                metadata=self.result
            )

        if path:
            messagebox.showinfo("Saved", f"Saved to:\n{path}", parent=self.window)