# Screenshot OCR Tool

A privacy-focused, fully offline desktop OCR tool for Windows. Capture any region of your screen and instantly extract visible text, no internet connection required, no data sent to external servers.

Built as a final year computing project (CI601) at the University of Brighton.

---

## Features

- **Visual region selector** : click and drag to select any area of your screen
- **Tesseract OCR** : accurate text extraction powered by the Tesseract LSTM engine
- **Image preprocessing** : greyscale conversion, noise reduction, Otsu binarisation and contrast enhancement to improve accuracy on degraded text
- **Live OCR visualiser** : watch bounding boxes drawn word by word, colour coded by confidence level
- **PII detection and redaction** : automatically detects email addresses, UK phone numbers, UK postcodes and credit card numbers, with redaction enabled by default
- **ML text categorisation** : classifies extracted text into Contact Information, Phone/Address, Code/Technical or Documentation using a pure Python TF-IDF and Naive Bayes classifier
- **Multiple export formats** : clipboard, TXT, DOCX and JSON
- **System tray application** : runs silently in the background, triggered via hotkey (default Ctrl+Shift+S) or tray right-click menu
- **Standalone EXE** : no Python or Tesseract installation required

---

## Download and Run

1. Go to the [Releases](../../releases) page
2. Download the latest `ScreenshotOCR.zip`
3. Extract the zip to any folder
4. Run `ScreenshotOCR.exe`

The application will minimise to the system tray. Right click the tray icon or press **Ctrl+Shift+S** to start a capture.

> **Windows only.** Tested on Windows 10 and Windows 11 at 100%, 125% and 150% display scaling.

---

## Usage

### Capturing text
1. Press **Ctrl+Shift+S** or right click the tray icon and select **Capture**
2. Click and drag to select the screen region containing the text
3. The live visualiser will show OCR processing in real time
4. The results window will open with the extracted text

### Exporting
From the results window you can:
- **Copy to Clipboard** — instant copy, no file created
- **Save As** — choose TXT, DOCX or JSON format and location

### Settings
Right click the tray icon and select **Settings** to configure:
- Enable or disable image preprocessing
- Redact PII by default
- Show or hide the live OCR visualiser
- Auto copy to clipboard after capture
- Change the capture hotkey

---

## Building from Source

### Requirements
- Python 3.13
- Tesseract OCR (portable installation in `tesseract/` folder)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run in development
```bash
python src/gui.py
```

### Build the executable
```bash
pyinstaller ScreenshotOCR.spec
```

The compiled output will be in `dist/ScreenshotOCR/`. Run `ScreenshotOCR.exe` from that folder.

> Note: UPX compression is enabled in the spec file. Download [UPX](https://upx.github.io/) and add it to your PATH to reduce the build size.

---

## Project Structure

```
src/
    gui.py                  Main application entry point
    screenshot.py           Region selector and screen capture
    ocr_engine.py           Tesseract OCR integration
    tesseract_manager.py    Tesseract path detection and lazy initialisation
    preprocessor.py         Image preprocessing pipeline
    export.py               TXT, DOCX, JSON and clipboard export
    categorization.py       Pure Python TF-IDF and Naive Bayes classifier
    pii_detection.py        Regex PII detection and redaction
    results_window.py       Post-capture results and export UI
    settings_window.py      Settings panel
    visualiser.py           Live OCR bounding box visualiser
    app_settings.py         Application settings dataclass
    main.py                 Console entry point (development only)
tesseract/
    tesseract.exe           Bundled Tesseract executable
    tessdata/               Language data files
ScreenshotOCR.spec          PyInstaller build configuration
requirements.txt            Python dependencies
```

---

## Dependencies

| Package | Purpose |
|---|---|
| pytesseract | Python wrapper for Tesseract OCR |
| Pillow | Screenshot capture and image preprocessing |
| python-docx | DOCX export |
| pyperclip | Clipboard copy |
| pystray | System tray icon |
| pynput | Global hotkey listener |

All ML classification (TF-IDF and Naive Bayes) is implemented using the Python standard library.

---

## Accuracy

| Test | Result |
|---|---|
| OCR accuracy (clean text) | 94% average confidence |
| OCR accuracy (degraded text, Arial font) | 93–95% |
| PII detection precision | 100% (20 samples, zero false positives) |
| Text categorisation accuracy | 80% (4 categories, 5 test samples) |
| Typical OCR processing time | 1.5–2.5 seconds |

---

## Privacy

All processing happens locally on your machine. No images, text or metadata are ever sent to any external server. PII redaction is enabled by default to protect sensitive information before export.

---

## License

MIT License — free to use, modify and distribute.

---

## Author

Harry Larkin — CI601 Computing Project, University of Brighton, 2026
