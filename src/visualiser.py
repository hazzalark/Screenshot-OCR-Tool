"""
Live OCR Visualiser
====================
Displays a Tkinter window over the captured screenshot and draws
confidence-coded bounding boxes word by word as Tesseract processes
the image. Provides real-time visual feedback of the OCR process.

Confidence colour coding:
    Green  >= 80%   High confidence
    Orange  60-79%  Medium confidence
    Red     < 60%   Low confidence

Author: Harry Larkin
Date: March 2026
"""

import threading
import traceback
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
import pytesseract

from ocr_engine import OCREngine


class LiveOCRVisualiser:
    """
    Fullscreen Tkinter overlay that animates OCR processing by drawing
    bounding boxes word by word as Tesseract recognises each word.

    OCR runs on a background daemon thread to avoid blocking the UI.
    The _tick() method is called every 40ms via root.after() and draws
    three words per tick, producing a smooth animated effect.

    The visualiser receives the OCREngine instance and calls
    _ensure_tesseract() before making any direct pytesseract calls,
    ensuring the Tesseract executable path is correctly configured
    whether the application is running in development or as a compiled EXE.

    Args:
        image_path:  Path to the captured screenshot PNG.
        on_complete: Callback fired with the OCR result dict when done.
        ocr_engine:  OCREngine instance used to initialise Tesseract path.
    """

    def __init__(self, image_path: str, on_complete, ocr_engine: OCREngine):
        self.image_path  = image_path
        self.on_complete = on_complete
        self.ocr_engine  = ocr_engine

        # Tkinter window and widget references
        self.root        = None
        self._canvas     = None
        self._status_var = None
        self._progress   = None

        # Image state
        self._draw_image = None   # PIL image being annotated with bounding boxes
        self._tk_image   = None   # PhotoImage reference kept to prevent GC
        self._disp_w     = 0      # Scaled display width
        self._disp_h     = 0      # Scaled display height

        # OCR state
        self._ocr_data   = None   # Raw Tesseract word-level data dict
        self._full_text  = ''     # Full extracted text string
        self._word_index = 0      # Current position in the word-by-word animation
        self._result     = None   # Final result dict passed to on_complete
        self._start_time = None   # Timestamp when OCR begins, used for processing time

        # Guard flag — prevents on_complete firing more than once
        # (e.g. if the after(2000) timer fires after the window is already closed)
        self._closed = False

    def start(self):
        """
        Open the visualiser window and start OCR on a background thread.
        Scales the image to fit the screen if necessary.
        """
        original         = Image.open(self.image_path)
        self._draw_image = original.copy()

        # Scale image to fit within 1200x800 without upscaling
        sw           = original.width
        sh           = original.height
        scale        = min(1200 / sw, 800 / sh, 1.0)
        self._disp_w = int(sw * scale)
        self._disp_h = int(sh * scale)

        # Create and configure the Toplevel window
        self.root = tk.Toplevel()
        self.root.title("OCR Processing")
        self.root.resizable(False, False)
        self.root.configure(bg='white')

        # Centre window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth()  - self._disp_w) // 2
        y = (self.root.winfo_screenheight() - self._disp_h - 70) // 2
        self.root.geometry(f"{self._disp_w}x{self._disp_h + 70}+{x}+{y}")
        self.root.lift()
        self.root.attributes('-topmost', True)

        # Bind ESC to cancel on both root and canvas (canvas needs focus_set)
        self.root.bind('<Escape>', lambda e: self._cancel())

        # Status bar showing current processing state
        self._status_var = tk.StringVar(value="Initialising OCR…")
        status_frame = tk.Frame(self.root, bg='#f0f0f0', pady=6)
        status_frame.pack(fill='x', side='top')

        tk.Label(
            status_frame,
            textvariable=self._status_var,
            bg='#f0f0f0',
            font=("Segoe UI", 9)
        ).pack(side='left', padx=10)

        # Confidence legend in the status bar
        legend = tk.Frame(status_frame, bg='#f0f0f0')
        legend.pack(side='right', padx=10)
        for colour, label in [("green", "High"), ("orange", "Medium"), ("red", "Low")]:
            tk.Label(legend, text="■", fg=colour,
                     bg='#f0f0f0', font=("Segoe UI", 9)).pack(side='left')
            tk.Label(legend, text=f"{label}  ",
                     bg='#f0f0f0', font=("Segoe UI", 9)).pack(side='left')

        # Progress bar tracking words processed vs total words
        self._progress = ttk.Progressbar(self.root, maximum=100, mode='determinate')
        self._progress.pack(fill='x')

        # Canvas for rendering the annotated screenshot
        self._canvas = tk.Canvas(
            self.root,
            width=self._disp_w,
            height=self._disp_h,
            bg='white',
            highlightthickness=0
        )
        self._canvas.pack()

        # Bind ESC on canvas and give it focus so keystrokes register
        self._canvas.bind('<Escape>', lambda e: self._cancel())
        self._canvas.focus_set()

        # Draw initial image before OCR starts
        self._update_canvas()

        # Start OCR on a daemon thread so the UI remains responsive
        threading.Thread(target=self._run_ocr, daemon=True).start()

        # Begin the animation tick loop
        self.root.after(100, self._tick)

    def _cancel(self):
        """
        Handle ESC key press — close the visualiser without triggering
        on_complete. Sets _closed so the after() timer cannot fire it later.
        """
        self._closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def _run_ocr(self):
        """
        Background thread: initialise Tesseract then run OCR on the image.

        Calls _ensure_tesseract() first so the executable path and
        TESSDATA_PREFIX are correctly set before any pytesseract calls.
        Uses PSM 3 (automatic page segmentation) and OEM 1 (LSTM only)
        for best accuracy on screen captures.
        """
        try:
            import time
            self._start_time = time.time()

            # Initialise Tesseract path via OCREngine before any direct calls
            self._status_var.set("Initialising Tesseract…")
            self.ocr_engine._ensure_tesseract()

            self._status_var.set("Running OCR…")
            original = Image.open(self.image_path)

            # Get word-level data with bounding boxes and confidence scores
            self._ocr_data = pytesseract.image_to_data(
                original,
                config='--psm 3 --oem 1',
                output_type=pytesseract.Output.DICT
            )

            # Get the full text string separately for the result dict
            self._full_text = pytesseract.image_to_string(
                original,
                config='--psm 3 --oem 1'
            )
            self._status_var.set("Drawing results…")

        except Exception as e:
            print(f"OCR error: {e}\n{traceback.format_exc()}")
            self._status_var.set(f"Error: {e}")
            self._ocr_data = None

    def _tick(self):
        """
        Animation tick called every 40ms via root.after().
        Draws three words per tick onto the image and updates the canvas.
        Calls _finish() once all words have been processed.
        """
        # Wait if OCR thread hasn't returned data yet
        if self._ocr_data is None:
            self.root.after(100, self._tick)
            return

        data  = self._ocr_data
        total = len(data['text'])
        draw  = ImageDraw.Draw(self._draw_image)

        # Process next batch of three words
        end = min(self._word_index + 3, total)

        for i in range(self._word_index, end):
            conf = int(data['conf'][i])
            word = data['text'][i].strip()

            # Skip words with no confidence score or empty text
            if conf <= 0 or not word:
                continue

            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]

            # Colour-code bounding box by confidence level
            colour = "green" if conf >= 80 else "orange" if conf >= 60 else "red"

            draw.rectangle([(x, y), (x + w, y + h)], outline=colour, width=2)
            draw.text((x, max(0, y - 12)), f"{conf}%", fill=colour)

        self._word_index = end

        # Update progress bar and status label
        self._progress['value'] = (end / total * 100) if total else 100
        words_done = sum(1 for t in data['text'][:end] if t.strip())
        self._status_var.set(f"Processing… {words_done} words recognised")

        self._update_canvas()

        if self._word_index < total:
            # Schedule next tick
            self.root.after(40, self._tick)
        else:
            self._finish()

    def _finish(self):
        """
        Called when all words have been drawn. Assembles the final result
        dict and schedules the window to close after 2 seconds.
        """
        import time

        self._progress['value'] = 100
        data            = self._ocr_data
        confs           = [int(c) for c in data['conf'] if int(c) != -1]
        avg_conf        = sum(confs) / len(confs) if confs else 0
        text            = self._full_text
        processing_time = round(time.time() - self._start_time, 2) if self._start_time else 0.0

        self._status_var.set(
            f"Done — {len(text.split())} words, "
            f"{avg_conf:.0f}% confidence, "
            f"{processing_time:.2f}s. Closing…"
        )

        # Build the result dict in the same format as OCREngine.extract_text_from_file
        self._result = {
            'text':               text,
            'confidence':         avg_conf,
            'processing_time':    processing_time,
            'char_count':         len(text.strip()),
            'word_count':         len(text.split()),
            'preprocessing_used': True,
        }

        # Close after 2 seconds so the user can read the completion message
        self.root.after(2000, self._close)

    def _close(self):
        """
        Close the window and fire on_complete exactly once.
        The _closed guard prevents double-firing if the window is destroyed
        before the after() timer fires.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self.root.destroy()
        except Exception:
            pass
        if self._result and self.on_complete:
            self.on_complete(self._result)

    def _update_canvas(self):
        """
        Refresh the canvas with the current state of _draw_image.
        Resizes to display dimensions before rendering.
        """
        display = self._draw_image.resize(
            (self._disp_w, self._disp_h),
            Image.Resampling.LANCZOS
        )
        # Keep a reference to prevent garbage collection
        self._tk_image = ImageTk.PhotoImage(display)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor='nw', image=self._tk_image)