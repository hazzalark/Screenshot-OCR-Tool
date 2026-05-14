"""
Image Preprocessor
===================
Applies a configurable image preprocessing pipeline to improve OCR
accuracy before passing images to Tesseract.

Pipeline steps (in recommended order per Tesseract documentation):
    1. Grayscale conversion  — reduces complexity, preserves text contrast
    2. Noise reduction       — median filter removes pixel-level noise
    3. Binarisation          — Otsu adaptive thresholding (pure Python)
    4. Contrast enhancement  — amplifies text-to-background contrast
    5. Deskewing             — optional rotation correction (disabled by default)

Research basis:
    Rice et al. (1996) and Mithe et al. (2013) identified image quality
    factors as the primary cause of OCR accuracy degradation. Preprocessing
    following the Tesseract-recommended sequence improves accuracy by
    15-25% on degraded inputs.

    The Otsu binarisation implementation follows the algorithm described in
    Otsu (1979), selecting the threshold that maximises between-class variance.
    Implemented in pure Python using Pillow's histogram() — no numpy required.

Author: Harry Larkin
Date: January 2026
"""

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract


class ImagePreprocessor:
    """
    Configurable image preprocessing pipeline for OCR accuracy improvement.

    Each step can be independently enabled or disabled via the config dict.
    The pipeline is designed to be applied to screenshots before passing
    them to Tesseract.

    All processing uses Pillow only — no numpy dependency.
    """

    def __init__(self):
        """
        Initialise the preprocessor with all steps enabled except deskewing.
        Deskewing is disabled by default due to its computational cost relative
        to the benefit on clean screen captures.
        """
        self.enabled = True

        # Individual step toggles — can be overridden via configure()
        self.config = {
            'grayscale':            True,
            'noise_reduction':      True,
            'binarization':         True,
            'contrast_enhancement': True,
            'deskew':               False,  # Expensive — opt-in only
        }

    def configure(self, **kwargs):
        """
        Enable or disable individual preprocessing steps.

        Args:
            grayscale (bool):            Convert image to grayscale
            noise_reduction (bool):      Apply median noise filter
            binarization (bool):         Apply Otsu adaptive binarisation
            contrast_enhancement (bool): Amplify contrast
            deskew (bool):               Correct text skew via Tesseract OSD

        Example:
            preprocessor.configure(deskew=True, noise_reduction=False)
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            else:
                print(f"Warning: unknown preprocessing option '{key}'")

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def process(self, image: Image.Image) -> Image.Image:
        """
        Apply the full preprocessing pipeline to the input image.

        Returns the image unchanged if preprocessing is disabled globally.
        Each enabled step is applied in the sequence recommended by the
        Tesseract documentation (Tesseract Wiki, 2023).

        Args:
            image: PIL Image to preprocess (any mode).

        Returns:
            Preprocessed PIL Image ready for Tesseract.
        """
        if not self.enabled:
            return image

        processed = image.copy()

        # Step 1 — Grayscale conversion
        # Reduces the image to a single luminance channel using the standard
        # ITU-R BT.601 formula: Grey = 0.299R + 0.587G + 0.114B
        # (Gonzalez & Woods, 2008). Simplifies all subsequent processing.
        if self.config['grayscale'] and processed.mode != 'L':
            processed = processed.convert('L')
            print("  Preprocessor: grayscale applied")

        # Step 2 — Noise reduction
        # Median filter replaces each pixel with the median of its neighbours,
        # effectively removing random noise spikes without blurring text edges.
        if self.config['noise_reduction']:
            processed = processed.filter(ImageFilter.MedianFilter(size=3))
            print("  Preprocessor: noise reduction applied")

        # Step 3 — Binarisation via Otsu's method
        # Converts the image to pure black and white by selecting the threshold
        # that maximises between-class variance (Otsu, 1979). The threshold is
        # calculated adaptively from the image histogram rather than being
        # hardcoded, making it robust to varying lighting conditions.
        if self.config['binarization']:
            threshold = self._otsu_threshold(processed)
            processed = processed.point(lambda p: 255 if p > threshold else 0)
            print(f"  Preprocessor: binarisation applied (threshold: {threshold})")

        # Step 4 — Contrast enhancement
        # Amplifies the difference between light and dark pixels, making text
        # more distinct from background regions that survived binarisation.
        if self.config['contrast_enhancement']:
            enhancer  = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(2.0)
            print("  Preprocessor: contrast enhancement applied")

        # Step 5 — Deskewing (optional)
        # Uses Tesseract's OSD (orientation and script detection) to detect
        # and correct text rotation. Disabled by default — expensive and
        # rarely needed for screen captures which are always axis-aligned.
        if self.config['deskew']:
            processed = self._deskew(processed)

        return processed

    # ── Otsu's thresholding ────────────────────────────────────────────────────

    def _otsu_threshold(self, image: Image.Image) -> int:
        """
        Calculate the optimal binarisation threshold using Otsu's method (1979).

        The algorithm exhaustively searches all 256 possible thresholds and
        selects the one that maximises between-class variance — the variance
        between the foreground (text) and background pixel populations.

        Uses Pillow's built-in histogram() to avoid a numpy dependency.
        Mathematically equivalent to the numpy implementation.

        Args:
            image: Grayscale PIL Image (mode 'L').

        Returns:
            Integer threshold value in range 0–255.
        """
        # Get the 256-bin pixel frequency histogram from Pillow
        hist  = image.histogram()
        total = sum(hist)

        if total == 0:
            return 128  # Fallback for empty images

        # Normalise histogram to probability distribution
        hist_norm = [count / total for count in hist]

        # Compute cumulative sum (w) and cumulative mean (m) in a single pass
        cum_sum, cum_mean = [], []
        cs, cm = 0.0, 0.0
        for i, prob in enumerate(hist_norm):
            cs += prob
            cm += prob * i
            cum_sum.append(cs)
            cum_mean.append(cm)

        # Global mean intensity used to compute the background class mean
        global_mean    = cum_mean[-1]
        best_threshold = 0
        best_variance  = 0.0

        # Find the threshold that maximises between-class variance:
        #   σ²_B(t) = w₀(t) · w₁(t) · (μ₀(t) − μ₁(t))²
        for t in range(256):
            w0 = cum_sum[t]           # Foreground class weight
            w1 = 1.0 - w0             # Background class weight

            # Skip degenerate cases where one class is empty
            if w0 == 0.0 or w1 == 0.0:
                continue

            m0       = cum_mean[t] / w0                         # Foreground mean
            m1       = (global_mean - cum_mean[t]) / w1         # Background mean
            variance = w0 * w1 * (m0 - m1) ** 2                 # Between-class variance

            if variance > best_variance:
                best_variance  = variance
                best_threshold = t

        return best_threshold

    # ── Deskewing ──────────────────────────────────────────────────────────────

    def _deskew(self, image: Image.Image) -> Image.Image:
        """
        Detect and correct text rotation using Tesseract's OSD engine.

        Requires Tesseract to be already initialised. Disabled by default
        due to the additional processing time.

        Args:
            image: Grayscale PIL Image to deskew.

        Returns:
            Rotated PIL Image, or the original if deskewing fails or is
            not needed.
        """
        try:
            # OSD returns orientation metadata including rotation angle
            osd   = pytesseract.image_to_osd(image)
            angle = 0

            for line in osd.split('\n'):
                if 'Rotate:' in line:
                    angle = int(line.split(':')[1].strip())
                    break

            if angle != 0:
                image = image.rotate(angle, expand=True, fillcolor='white')
                print(f"  Preprocessor: deskewed {angle} degrees")

            return image

        except Exception as e:
            # OSD can fail on images with insufficient text — silently skip
            print(f"  Preprocessor: deskew skipped ({e})")
            return image