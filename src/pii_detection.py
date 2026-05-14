"""
PII Detection and Redaction Module
====================================
Detects and redacts Personally Identifiable Information (PII) from
OCR-extracted text using regex pattern matching.

Supported PII types:
    - Email addresses    (RFC 5322 simplified pattern)
    - UK phone numbers   (mobile, landline, freephone, +44 international)
    - UK postcodes       (all valid formats)
    - Credit card numbers (Visa, Mastercard, Amex, Discover)

Detection approach:
    Regex pattern matching is used for all four types. As documented in
    Narayanan and Shmatikov (2008), pattern matching achieves >95% precision
    on well-defined structured formats, making it the appropriate approach
    for these data types without the overhead of a Named Entity Recognition
    model.

    Full address parsing was evaluated but excluded. Structured address
    detection produces too many false positives on text with similar
    formatting (numbered lists, document headings). UK postcodes are
    included as a reliable proxy — their presence strongly implies an
    address context.

Redaction behaviour:
    Following the principle of privacy by design (Cavoukian, 2009),
    redaction is enabled by default. Detected PII is replaced with
    clearly labelled tags e.g. [REDACTED:EMAIL ADDRESS]. Users can
    choose to keep the original text before exporting.

Author: Harry Larkin
Date: January 2026
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# ── PII type registry ──────────────────────────────────────────────────────────
# Maps internal keys to human-readable labels used in redaction tags
# and the results window summary.

PII_TYPES = {
    'email':       'Email Address',
    'phone':       'UK Phone Number',
    'postcode':    'UK Postcode',
    'credit_card': 'Credit Card Number',
}


# ── Regex patterns ─────────────────────────────────────────────────────────────

# Email addresses — simplified RFC 5322 pattern covering all common formats.
# Matches local-part@domain.tld with standard special characters in the
# local part (dots, underscores, percent, plus, hyphens).
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
)

# UK phone numbers — four sub-patterns joined by alternation (|):
#   1. Mobile:    07xxx xxxxxx or +44 7xxx xxxxxx
#   2. Landline:  01xxx xxxxxx, 02x xxxx xxxx, or +44 variants
#   3. Freephone: 0800, 0845, 0870, 0808 formats
# Separators between digit groups are optional and may be spaces, hyphens
# or dots to handle varied formatting in OCR output.
PHONE_PATTERN = re.compile(
    r'(\+44\s?7\d{3}|\(?07\d{3}\)?)'       # Mobile: +44 7xxx or 07xxx
    r'[\s\-]?\d{3}[\s\-]?\d{3}'
    r'|(\+44\s?\d{2,4}|\(?0\d{3,4}\)?)'    # Landline: +44 or 0x variants
    r'[\s\-]?\d{3,4}[\s\-]?\d{3,4}'
    r'|(0800|0845|0870|0808)'               # Freephone and premium rate
    r'[\s\-]?\d{3}[\s\-]?\d{4}'
)

# UK postcodes — covers all six valid outward code formats:
#   A9 9AA, A99 9AA, AA9 9AA, AA99 9AA, AA9A 9AA, A9A 9AA
# Case-insensitive to handle OCR output that may not preserve case.
POSTCODE_PATTERN = re.compile(
    r'\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s?(\d[A-Z]{2})\b',
    re.IGNORECASE
)

# Credit card numbers — four card scheme patterns joined by alternation:
#   Visa:       16 digits starting with 4  (4xxx xxxx xxxx xxxx)
#   Mastercard: 16 digits starting with 51-55
#   Amex:       15 digits in 4-6-5 format  (34xx or 37xx)
#   Discover:   16 digits starting with 6011 or 65xx
# Digit groups may be separated by spaces or hyphens as OCR output varies.
# Note: Luhn algorithm validation is not applied — pattern matching only.
CREDIT_CARD_PATTERN = re.compile(
    r'\b(?:'
    r'4\d{3}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}'           # Visa 16-digit
    r'|5[1-5]\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}'     # Mastercard 16-digit
    r'|3[47]\d{2}[\s\-]?\d{6}[\s\-]?\d{5}'                   # Amex 15-digit
    r'|6(?:011|5\d{2})[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}' # Discover 16-digit
    r')\b'
)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class PIIMatch:
    """
    Represents a single detected PII instance within a text string.

    Attributes:
        pii_type: Internal key (e.g. 'email', 'phone').
        label:    Human-readable label (e.g. 'Email Address').
        value:    The exact matched text from the input.
        start:    Start character index in the original text.
        end:      End character index in the original text.
        redacted: Whether this instance should be redacted on export.
    """
    pii_type: str
    label:    str
    value:    str
    start:    int
    end:      int
    redacted: bool = True


@dataclass
class PIIResult:
    """
    Encapsulates the full result of a PII detection pass on a text string.

    Attributes:
        original_text: The input text before any redaction.
        redacted_text: Text with PII replaced by [REDACTED:TYPE] tags,
                       or the original text if redaction was not applied.
        matches:       All PIIMatch objects found in the text.
        redact:        Whether redaction was applied in this pass.
    """
    original_text: str
    redacted_text: str
    matches:       List[PIIMatch] = field(default_factory=list)
    redact:        bool = True

    @property
    def has_pii(self) -> bool:
        """True if any PII was detected in the text."""
        return len(self.matches) > 0

    @property
    def summary(self) -> str:
        """
        Short human-readable summary for the results window warning bar.

        Example: 'PII detected: 2x Email Address, 1x UK Phone Number'
        """
        if not self.has_pii:
            return "No PII detected."
        counts = {}
        for m in self.matches:
            counts[m.label] = counts.get(m.label, 0) + 1
        parts = [f"{count}x {label}" for label, count in counts.items()]
        return "PII detected: " + ", ".join(parts)

    @property
    def type_counts(self) -> Dict[str, int]:
        """Return a dict mapping PII type keys to match counts."""
        counts = {}
        for m in self.matches:
            counts[m.pii_type] = counts.get(m.pii_type, 0) + 1
        return counts


# ── Detector ───────────────────────────────────────────────────────────────────

class PIIDetector:
    """
    Detects and redacts PII from OCR-extracted text using regex patterns.

    Privacy by design (Cavoukian, 2009): redaction is enabled by default.
    Users can choose to keep the original text before exporting via the
    redact checkbox in the results window.

    Overlapping matches are resolved by keeping the longer match — for
    example, if a phone pattern fires inside a longer credit card match,
    the credit card match takes precedence.
    """

    def __init__(self, redact_by_default: bool = True):
        """
        Args:
            redact_by_default: Whether to redact automatically when process()
                               is called without an explicit redact argument.
                               Controlled by AppSettings.pii_redaction.
        """
        self.redact_by_default = redact_by_default

        # Mapping of type keys to compiled regex patterns
        self._patterns: Dict[str, re.Pattern] = {
            'email':       EMAIL_PATTERN,
            'phone':       PHONE_PATTERN,
            'postcode':    POSTCODE_PATTERN,
            'credit_card': CREDIT_CARD_PATTERN,
        }

    # ── Detection ──────────────────────────────────────────────────────────────

    def detect(self, text: str) -> List[PIIMatch]:
        """
        Scan text for all supported PII types and return sorted matches.

        Each compiled pattern is run against the full text. Results are
        collected, sorted by start position, then filtered to remove
        overlapping matches (longer match preferred).

        Args:
            text: The text to scan for PII.

        Returns:
            List of PIIMatch objects sorted by position.
            Empty list if no PII is detected or text is empty.
        """
        if not text or not text.strip():
            return []

        raw_matches: List[PIIMatch] = []

        # Run each regex pattern and collect all matches across the text
        for pii_type, pattern in self._patterns.items():
            for m in pattern.finditer(text):
                raw_matches.append(PIIMatch(
                    pii_type = pii_type,
                    label    = PII_TYPES[pii_type],
                    value    = m.group().strip(),
                    start    = m.start(),
                    end      = m.end(),
                ))

        # Sort all matches by start position
        raw_matches.sort(key=lambda x: x.start)

        # Remove overlapping matches — keep the longer match in each conflict
        filtered: List[PIIMatch] = []
        last_end = -1

        for match in raw_matches:
            if match.start >= last_end:
                # No overlap with previous match — include it
                filtered.append(match)
                last_end = match.end
            else:
                # Overlap — replace the previous match if this one is longer
                if filtered and (match.end - match.start) > (filtered[-1].end - filtered[-1].start):
                    filtered[-1] = match
                    last_end     = match.end

        return filtered

    # ── Redaction ──────────────────────────────────────────────────────────────

    def redact_text(self, text: str, matches: List[PIIMatch]) -> str:
        """
        Replace detected PII with [REDACTED:TYPE] tags.

        Matches are processed in reverse order so substituting one match
        does not shift the character indices of earlier matches.

        Args:
            text:    Original text string.
            matches: PIIMatch list from detect().

        Returns:
            Text with all redacted matches replaced by labelled tags.

        Example:
            Input:  'Email john@test.com or call 07700 900123'
            Output: 'Email [REDACTED:EMAIL ADDRESS] or call [REDACTED:UK PHONE NUMBER]'
        """
        if not matches:
            return text

        result = text

        # Reverse iteration preserves indices of earlier matches
        for match in reversed(matches):
            if match.redacted:
                tag    = f"[REDACTED:{match.label.upper()}]"
                result = result[:match.start] + tag + result[match.end:]

        return result

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(
        self,
        text:   str,
        redact: Optional[bool] = None,
    ) -> PIIResult:
        """
        Detect PII in text and optionally apply redaction.

        Primary method called by the capture pipeline after OCR completes.
        The redact argument allows per-call override of the default behaviour.

        Args:
            text:   OCR-extracted text to scan.
            redact: True to redact, False to skip, None to use the default.

        Returns:
            PIIResult with original text, redacted text and match list.
        """
        should_redact = redact if redact is not None else self.redact_by_default
        matches       = self.detect(text)

        redacted_text = (
            self.redact_text(text, matches)
            if should_redact and matches
            else text
        )

        return PIIResult(
            original_text = text,
            redacted_text = redacted_text,
            matches       = matches,
            redact        = should_redact,
        )

    def get_export_text(self, result: PIIResult, user_choice: str = 'redact') -> str:
        """
        Return the correct text for export based on the user's choice in the
        results window redact checkbox.

        Args:
            result:      PIIResult from process().
            user_choice: 'redact' returns redacted_text (default).
                         'keep'   returns original_text.

        Returns:
            Text string to pass to the exporter.
        """
        return result.original_text if user_choice == 'keep' else result.redacted_text

    def print_summary(self, result: PIIResult):
        """
        Print a formatted PII detection summary to the console.
        Detected values are partially masked (first/last 2 chars shown).
        """
        if not result.has_pii:
            print("PII scan: no PII detected.")
            return

        print(f"PII scan: {result.summary}")
        print(f"  {len(result.matches)} item(s) found:")

        for i, match in enumerate(result.matches, 1):
            # Partially mask value for safe console display
            val    = match.value
            masked = (
                val[:2] + ('*' * (len(val) - 4)) + val[-2:]
                if len(val) > 6 else '*' * len(val)
            )
            print(f"  {i}. [{match.label}] {masked}  "
                  f"(index {match.start}–{match.end})")


# ── Standalone test ────────────────────────────────────────────────────────────

def test_pii_detector():
    """Smoke test covering all four PII types and the no-PII case."""
    print("=" * 60)
    print("PII DETECTION TEST")
    print("=" * 60)

    detector = PIIDetector(redact_by_default=True)

    samples = [
        ("Email",
         "Please contact john.smith@example.co.uk for more information."),
        ("UK Phone",
         "Call us on 07700 900123 or 01273 456789 between 9am and 5pm."),
        ("UK Postcode",
         "Our office is at BN1 1AB and deliveries go to SW1A 2AA."),
        ("Credit Card",
         "Card number: 4111 1111 1111 1111 expiry 12/26 CVV 123"),
        ("Mixed PII",
         "Contact Sarah at sarah@company.com, mobile 07911 123456, "
         "based in EC1A 1BB. Invoice paid by card 5500 0000 0000 0004."),
        ("No PII",
         "This is a standard document with no personal information."),
    ]

    for label, text in samples:
        print(f"\n--- {label} ---")
        print(f"Input: {text[:80]}{'…' if len(text) > 80 else ''}")
        result = detector.process(text)
        detector.print_summary(result)
        if result.has_pii:
            print(f"Redacted: {result.redacted_text}")

    print("\n" + "=" * 60)
    print("Test complete.")


if __name__ == "__main__":
    test_pii_detector()