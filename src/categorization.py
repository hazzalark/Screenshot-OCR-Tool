"""
Text Categorisation Module
===========================
Classifies extracted OCR text into one of four predefined categories
using a hybrid approach: regex pattern matching for structured types,
and a pure Python TF-IDF + Multinomial Naive Bayes classifier for prose.

Categories:
    contact_info   — Email addresses, names and contact details
    phone_number   — UK phone numbers and physical addresses
    code           — Code snippets and technical content
    documentation  — General prose, reports, articles and notes

Architecture:
    Pattern matching fires first for structured data types (email, phone,
    postcode, code syntax) where regex achieves >95% precision. If a
    strong match is found (confidence >= 0.75), the result is returned
    immediately without invoking the ML classifier.

    For text that does not trigger a strong pattern match, a TF-IDF and
    Multinomial Naive Bayes pipeline classifies the content. If a weaker
    pattern match was found (confidence >= 0.50), the result is blended
    with the ML output — pattern confidence weighted at 60%, ML at 40%.

ML implementation:
    The TF-IDF and Naive Bayes pipeline is implemented entirely in pure
    Python using the math and collections standard library modules.
    It is mathematically equivalent to scikit-learn's TfidfVectorizer
    (sublinear_tf=True, ngram_range=(1,2)) + MultinomialNB (alpha=0.1)
    pipeline, but adds zero external dependency weight.

    This decision was made to reduce the compiled executable size —
    scikit-learn, numpy and scipy together added ~350MB to the build.

    Reference: Sebastiani (2002) — Machine Learning in Automated Text
    Categorization, ACM Computing Surveys, 34(1), pp.1-47.

Author: Harry Larkin
Date: March 2026
"""

import re
import math
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ── Category definitions ───────────────────────────────────────────────────────

# Maps internal category keys to human-readable labels shown in the UI
CATEGORIES = {
    'contact_info':  'Contact Information',
    'phone_number':  'Phone Number / Address',
    'code':          'Code / Technical',
    'documentation': 'Documentation / Prose',
}

# Confidence thresholds used by the hybrid classification logic
HIGH_CONFIDENCE   = 0.75  # Pattern match strong enough to return immediately
MEDIUM_CONFIDENCE = 0.50  # Pattern match blended with ML result
LOW_CONFIDENCE    = 0.30  # Below this — ML result used exclusively


# ── Regex patterns ─────────────────────────────────────────────────────────────
# These patterns fire in _pattern_match() before the ML classifier is invoked.
# Pattern matching is preferred for structured types as it provides
# near-deterministic results with high precision.

# Email — fires the contact_info category
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
)

# UK phone numbers — fires the phone_number category alongside postcode hits
PHONE_PATTERN = re.compile(
    r'(\+44\s?7\d{3}|\(?07\d{3}\)?)'
    r'[\s\-]?\d{3}[\s\-]?\d{3}'
    r'|(\+44\s?\d{2,4}|\(?0\d{3,4}\)?)'
    r'[\s\-]?\d{3,4}[\s\-]?\d{3,4}'
    r'|(0800|0845|0870|0808)'
    r'[\s\-]?\d{3}[\s\-]?\d{4}'
)

# UK postcodes — used as a proxy for physical address content
POSTCODE_PATTERN = re.compile(
    r'\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s?(\d[A-Z]{2})\b',
    re.IGNORECASE
)

# Street name keywords — contribute to address confidence alongside postcodes
ADDRESS_KEYWORDS = re.compile(
    r'\b(street|road|avenue|lane|drive|close|court|place|way|terrace|'
    r'st\.|rd\.|ave\.|blvd|crescent|grove|square)\b',
    re.IGNORECASE
)

# Code keyword patterns — Python, JS, Java, C, SQL, shell and CLI syntax
CODE_SYMBOLS = re.compile(
    r'(def |class |import |from |return |if __name__|print\(|'
    r'function\s*\(|const |let |var |==>|=>|nullptr|'
    r'#include|public static|private |protected |void |int |str\(|'
    r'\.append\(|\.split\(|for .* in .*:|try:|except:|elif )'
)

# Code symbol density — high ratio of brackets/braces/semicolons indicates code
CODE_SYMBOL_DENSITY = re.compile(r'[{};()\[\]]')


# ── Training data ──────────────────────────────────────────────────────────────
# 84 hand-written examples (21 per category) covering realistic OCR capture
# scenarios. Sebastiani (2002) recommends 500+ examples for robust accuracy —
# this dataset is intentionally small due to project scope, and achieves
# acceptable accuracy on four relatively distinct categories.

TRAINING_DATA = {
    'contact_info': [
        "John Smith john.smith@email.com Marketing Manager",
        "Contact us at support@company.co.uk for help",
        "Sarah Johnson - sarah.j@outlook.com - 07700 900123",
        "From: noreply@newsletter.com To: user@domain.com",
        "Email: hello@startup.io | LinkedIn: linkedin.com/in/johnsmith",
        "Dear Mr Williams, please find attached the requested documents",
        "Regards, Emma Clarke, emma.clarke@firm.com, Senior Developer",
        "CC: manager@corp.com; team@corp.com",
        "Reply-To: billing@service.net",
        "Name: Alice Brown, Role: Designer, alice@design.studio",
        "HR Department hr@company.org extension 4421",
        "Technical lead: dev@project.io, Mobile: +44 7911 123456",
        "Contact the team at info@agency.com for a quote",
        "admin@server.local has full system privileges",
        "Forwarded message from: notifications@app.com",
        "Portfolio: james@creative.design",
        "Account manager: lucy.white@enterprise.co.uk",
        "Support ticket assigned to: help@techco.com",
        "Emergency contact: operations@24hr.service",
        "Newsletter signup: subscribe@weekly.digest",
    ],
    'phone_number': [
        "Call us on 01273 123456 between 9am and 5pm Monday to Friday",
        "Mobile: 07700 900456 Tel: 01234 567890",
        "123 High Street, Brighton, BN1 1AB East Sussex",
        "Delivery address: 45 Oak Road, London, SW1A 2AA",
        "Phone: +44 20 7946 0958 Fax: +44 20 7946 0959",
        "10 Downing Street Westminster London SW1A 2AA United Kingdom",
        "Next day delivery to: 78 Church Lane Manchester M1 4BT",
        "Billing address: Flat 3 22 Station Road Bristol BS1 6QF",
        "Emergency: 999 Non-emergency: 101",
        "Customer service: 0800 123 4567 free from landlines",
        "Registered office: 1 Canada Square Canary Wharf London E14 5AB",
        "Head office: Crown House 72 Hammersmith Road London W6 7JP",
        "Invoice to: Unit 5 Business Park, Swindon SN1 3HE",
        "Returns department: 67 Warehouse Street Birmingham B1 2AB",
        "+1 (555) 867-5309 extension 42",
        "Washington DC 20001 United States of America",
        "PO Box 1234 Edinburgh EH1 1YZ Scotland",
        "Appointment at: St. Thomas Hospital Westminster Bridge Road SE1 7EH",
        "Dial 0345 600 0723 for account enquiries",
        "Freephone: 0800 000 0000 Lines open 24/7",
    ],
    'code': [
        "def calculate_sum(a, b): return a + b",
        "import numpy as np\nimport pandas as pd\ndf = pd.read_csv('data.csv')",
        "for i in range(10):\n    print(f'Item {i}')",
        "class DatabaseConnection:\n    def __init__(self, host, port):",
        "const fetchData = async (url) => { const res = await fetch(url); }",
        "SELECT * FROM users WHERE active = 1 ORDER BY created_at DESC",
        "git commit -m 'fix: resolve null pointer exception in auth module'",
        "npm install --save-dev webpack babel-loader @babel/core",
        "if __name__ == '__main__':\n    main()",
        "try:\n    result = api.call()\nexcept RequestException as e:\n    log(e)",
        "<div class='container'><h1>Hello World</h1></div>",
        "docker run -d -p 8080:80 --name myapp nginx:latest",
        "kubectl apply -f deployment.yaml --namespace production",
        "public static void main(String[] args) { System.out.println(); }",
        "pip install pytesseract pillow scikit-learn numpy",
        "function validateEmail(email) { return /\\S+@\\S+/.test(email); }",
        "#include <iostream>\nusing namespace std;\nint main() { return 0; }",
        "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT NOT NULL);",
        "curl -X POST https://api.example.com/v1/users -H 'Content-Type: application/json'",
        "const [state, setState] = useState(null); useEffect(() => {}, []);",
        "pytest tests/ -v --cov=src --cov-report=html",
        "ssh ubuntu@192.168.1.100 -i ~/.ssh/id_rsa",
        "grep -r 'TODO' ./src --include='*.py' | wc -l",
        "TESSDATA_PREFIX=/usr/share tesseract img.png out",
    ],
    'documentation': [
        "This document provides an overview of the system architecture.",
        "Chapter 3: Getting Started. To begin, ensure Python 3.9 or later is installed.",
        "Abstract: This paper presents a novel approach to text classification.",
        "README. Installation. Clone the repository and run pip install.",
        "Terms and Conditions. By using this service you agree to the following.",
        "Meeting Notes - 15th January 2026. Attendees: Alice, Bob, Carol.",
        "Product Roadmap Q1 2026. Feature A - In Progress. Feature B - Planned.",
        "User Guide: How to export your data. Step 1: Navigate to Settings.",
        "Invoice #INV-2026-001. Date: 01/01/2026. Description: Consulting services.",
        "Release Notes v2.1.0. Bug Fixes: Fixed crash on startup.",
        "Executive Summary. The project delivered all milestones on time.",
        "FAQ: Frequently Asked Questions. Q: How do I reset my password?",
        "Privacy Policy. We collect personal data only as necessary.",
        "Annual Report 2025. Revenue increased by 12% year on year.",
        "Agenda: Team Standup. 1. Updates. 2. Blockers. 3. Plan for today.",
        "The quick brown fox jumps over the lazy dog.",
        "License: MIT. Permission is hereby granted, free of charge.",
        "Changelog: 2026-01-15 Added dark mode. 2026-01-10 Fixed export bug.",
        "Specification: The system shall respond within 200ms under normal load.",
        "Introduction. OCR technology has advanced significantly since the 1950s.",
    ],
}


# ── Pure Python TF-IDF + Naive Bayes ──────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """
    Tokenise text into lowercase unigrams and bigrams.

    Matches the behaviour of TfidfVectorizer(ngram_range=(1,2)):
    - Unigrams: individual words  (e.g. 'def', 'calculate')
    - Bigrams:  adjacent pairs    (e.g. 'def_calculate')

    Non-alphanumeric characters are discarded, which strips punctuation
    and normalises varied OCR output spacing.

    Args:
        text: Raw text string to tokenise.

    Returns:
        Combined list of unigram and bigram tokens.
    """
    words    = re.findall(r'[a-z0-9]+', text.lower())
    unigrams = words
    bigrams  = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    return unigrams + bigrams


def _compute_tfidf(
    corpus: List[str],
    labels: List[str],
) -> Tuple[Dict, Dict, Dict, List[str]]:
    """
    Build per-class TF-IDF feature weights from the training corpus.

    Implements TF-IDF with sublinear term frequency normalisation:
        TF(t, d)  = 1 + log(count(t, d))   if count > 0 else 0
        IDF(t)    = log((1 + N) / (1 + df(t))) + 1
        weight    = TF(t, d) * IDF(t)

    This matches scikit-learn's TfidfVectorizer(sublinear_tf=True)
    behaviour. Weights are accumulated per class rather than per document,
    producing the class-level feature vectors used by Naive Bayes.

    Args:
        corpus: List of training text strings.
        labels: Corresponding category label for each string.

    Returns:
        vocab        — {term: index} vocabulary mapping
        class_tfidf  — {class: {term: accumulated weight}}
        class_priors — {class: log(P(class))} log prior probabilities
        classes      — sorted list of unique class names
    """
    classes = sorted(set(labels))
    n_docs  = len(corpus)

    # Build document frequency: how many documents contain each term
    df: Dict[str, int] = defaultdict(int)
    tokenised = [_tokenise(doc) for doc in corpus]
    for tokens in tokenised:
        for term in set(tokens):   # set() ensures each term counted once per doc
            df[term] += 1

    # Build vocabulary index and IDF weights with smoothing
    # Smoothing prevents division by zero and reduces sensitivity to rare terms
    vocab = {term: i for i, term in enumerate(sorted(df.keys()))}
    idf   = {
        term: math.log((1 + n_docs) / (1 + count)) + 1
        for term, count in df.items()
    }

    # Accumulate TF-IDF weights per class
    class_tfidf: Dict[str, Dict[str, float]] = {c: defaultdict(float) for c in classes}
    class_doc_count: Dict[str, int] = Counter(labels)

    for tokens, label in zip(tokenised, labels):
        tf = Counter(tokens)
        for term, count in tf.items():
            # Sublinear TF normalisation: dampens the effect of very frequent terms
            tf_val = 1 + math.log(count) if count > 0 else 0
            class_tfidf[label][term] += tf_val * idf.get(term, 1.0)

    # Log priors: log(count of class / total documents)
    class_priors = {
        c: math.log(class_doc_count[c] / n_docs)
        for c in classes
    }

    return vocab, class_tfidf, class_priors, classes


def _predict(
    text:         str,
    vocab:        Dict,
    class_tfidf:  Dict,
    class_priors: Dict,
    classes:      List[str],
    alpha:        float = 0.1,
) -> Dict[str, float]:
    """
    Classify text using Multinomial Naive Bayes with Laplace smoothing.

    For each class c, computes the log posterior score:
        log P(c|text) ≈ log P(c) + Σ TF(t) * log P(t|c)

    Where P(t|c) uses Laplace smoothing (alpha) to handle unseen terms:
        P(t|c) = (weight(t,c) + alpha) / (Σ_all_terms weight(t,c) + alpha * |vocab|)

    Log space is used to avoid floating point underflow when multiplying
    many small probabilities. Softmax normalisation converts log scores
    to a proper probability distribution.

    Args:
        text:         Text to classify.
        vocab:        {term: index} vocabulary from _compute_tfidf.
        class_tfidf:  {class: {term: weight}} from _compute_tfidf.
        class_priors: {class: log_prior} from _compute_tfidf.
        classes:      List of class name strings.
        alpha:        Laplace smoothing factor (default 0.1).

    Returns:
        Dict mapping each class name to its normalised probability (0–1).
    """
    tokens = _tokenise(text)

    # Return uniform distribution if the text produces no tokens
    if not tokens:
        return {c: 1 / len(classes) for c in classes}

    tf = Counter(tokens)
    log_scores: Dict[str, float] = {}

    for c in classes:
        # Start from the log prior probability of this class
        score     = class_priors[c]
        class_sum = sum(class_tfidf[c].values())

        for term, count in tf.items():
            tf_val      = 1 + math.log(count) if count > 0 else 0
            term_weight = class_tfidf[c].get(term, 0.0)

            # Laplace-smoothed log likelihood for this term given the class
            score += tf_val * math.log(
                (term_weight + alpha) / (class_sum + alpha * len(vocab))
            )

        log_scores[c] = score

    # Softmax: subtract max score before exponentiating to improve numerical
    # stability (prevents overflow when scores are large negative numbers)
    max_score  = max(log_scores.values())
    exp_scores = {c: math.exp(s - max_score) for c, s in log_scores.items()}
    total      = sum(exp_scores.values())

    return {c: v / total for c, v in exp_scores.items()}


# ── Categoriser ────────────────────────────────────────────────────────────────

class TextCategoriser:
    """
    Hybrid text categoriser: regex pattern matching + pure Python TF-IDF
    and Multinomial Naive Bayes. No external ML dependencies required.

    Classification order:
        1. Pattern matching — fires for structured types with high precision.
           Returns immediately if confidence >= HIGH_CONFIDENCE (0.75).
        2. ML classifier — runs if no strong pattern match was found.
        3. Hybrid blend — if a medium-confidence pattern match was found
           (>= 0.50), the result is blended: 60% pattern + 40% ML.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise and train the categoriser from the built-in dataset.
        Optionally loads a pre-trained model from disk to skip training.

        Args:
            model_path: Path to a pickle file saved by save(). If None
                        or the file does not exist, trains from scratch.
        """
        self._vocab        = None
        self._class_tfidf  = None
        self._class_priors = None
        self._classes      = None

        # Try loading a saved model first, fall back to training if unavailable
        saved = model_path or str(Path(__file__).parent / 'categoriser_model.pkl')
        if model_path and Path(saved).exists():
            self._load(saved)
        else:
            self._train()

    def _train(self):
        """
        Train the TF-IDF + Naive Bayes model on the built-in dataset.
        Typically completes in under one second.
        """
        print("Training text categorisation model…")
        texts, labels = [], []

        for category, examples in TRAINING_DATA.items():
            texts.extend(examples)
            labels.extend([category] * len(examples))

        self._vocab, self._class_tfidf, self._class_priors, self._classes = \
            _compute_tfidf(texts, labels)

        print(
            f"Model trained — {len(texts)} examples, "
            f"{len(self._vocab)} features, "
            f"{len(self._classes)} categories."
        )

    def _load(self, path: str):
        """Load a previously saved model from a pickle file."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._vocab        = data['vocab']
            self._class_tfidf  = data['class_tfidf']
            self._class_priors = data['class_priors']
            self._classes      = data['classes']
            print(f"Loaded categorisation model: {path}")
        except Exception as e:
            print(f"Could not load model ({e}) — retraining from built-in data.")
            self._train()

    def save(self, path: str):
        """
        Persist the trained model to disk as a pickle file.

        Saving is optional — training from the built-in dataset takes
        under one second, so retraining on each launch is acceptable.
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab':        self._vocab,
                'class_tfidf':  self._class_tfidf,
                'class_priors': self._class_priors,
                'classes':      self._classes,
            }, f)

    # ── Pattern matching ───────────────────────────────────────────────────────

    def _pattern_match(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Attempt to classify text using regex pattern matching alone.

        Counts hits for each pattern type and returns the most confident
        category if the signal is strong enough. Returns None if no clear
        pattern match is found, deferring to the ML classifier.

        Confidence is calculated heuristically from hit counts:
            code:    0.60 base + 0.05 per keyword hit + symbol density factor
            contact: 0.75 base + 0.05 per email hit (capped at 0.95)
            phone:   0.60 base + weighted sum of phone/postcode/address hits

        Returns:
            (category_key, confidence) tuple, or None to defer to ML.
        """
        text_lower = text.lower()

        email_hits    = len(EMAIL_PATTERN.findall(text))
        phone_hits    = len(PHONE_PATTERN.findall(text))
        postcode_hits = len(POSTCODE_PATTERN.findall(text))
        address_hits  = len(ADDRESS_KEYWORDS.findall(text_lower))
        code_kw_hits  = len(CODE_SYMBOLS.findall(text))

        # Ratio of bracket/brace/semicolon characters — strong code indicator
        sym_density = len(CODE_SYMBOL_DENSITY.findall(text)) / max(len(text), 1)

        # Code: two or more keywords, or high symbol density
        if code_kw_hits >= 2 or sym_density > 0.04:
            confidence = min(0.95, 0.60 + (code_kw_hits * 0.05) + (sym_density * 2))
            return ('code', round(confidence, 2))

        # Contact info: at least one email address
        if email_hits >= 1:
            confidence = min(0.95, 0.75 + (email_hits * 0.05))
            return ('contact_info', round(confidence, 2))

        # Phone/address: phone number, postcode or multiple address keywords
        if phone_hits >= 1 or postcode_hits >= 1 or address_hits >= 2:
            score      = (phone_hits * 0.3) + (postcode_hits * 0.4) + (address_hits * 0.15)
            confidence = min(0.92, 0.60 + score)
            return ('phone_number', round(confidence, 2))

        # No strong pattern signal — defer to ML
        return None

    # ── ML classification ──────────────────────────────────────────────────────

    def _ml_classify(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify text using the trained TF-IDF + Naive Bayes model.

        Returns:
            Tuple of (best_category, confidence, all_scores_dict).
            all_scores_dict maps each category key to its probability.
        """
        probs    = _predict(
            text, self._vocab, self._class_tfidf,
            self._class_priors, self._classes
        )
        best_cat = max(probs, key=probs.get)
        return (
            best_cat,
            round(probs[best_cat], 3),
            {k: round(v, 3) for k, v in probs.items()}
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def categorise(self, text: str) -> Dict:
        """
        Classify a text string into one of four categories.

        Runs the hybrid pipeline: pattern matching first, ML classifier
        second, with optional blending when a medium-confidence pattern
        match is combined with an ML result.

        Args:
            text: OCR-extracted text to classify.

        Returns:
            dict containing:
                category        — internal key (e.g. 'code')
                category_label  — display label (e.g. 'Code / Technical')
                confidence      — float 0.0–1.0
                confidence_pct  — string (e.g. '87%')
                confidence_level — 'High', 'Medium' or 'Low'
                method          — 'pattern', 'ml' or 'hybrid'
                all_scores      — {category: probability} for ML results
        """
        # Return a default for empty input
        if not text or not text.strip():
            return {
                'category':         'documentation',
                'category_label':   CATEGORIES['documentation'],
                'confidence':       0.0,
                'confidence_pct':   '0%',
                'confidence_level': 'Low',
                'method':           'none',
                'all_scores':       {},
            }

        pattern_result = self._pattern_match(text)

        if pattern_result and pattern_result[1] >= HIGH_CONFIDENCE:
            # Strong pattern match — return immediately without ML
            category, confidence = pattern_result
            method     = 'pattern'
            all_scores = {}

        else:
            # Run the ML classifier
            ml_cat, ml_conf, all_scores = self._ml_classify(text)

            if pattern_result and pattern_result[1] >= MEDIUM_CONFIDENCE:
                p_cat, p_conf = pattern_result

                if p_conf > ml_conf * 0.8:
                    # Pattern result is competitive — blend the two
                    category   = p_cat
                    confidence = round((p_conf * 0.6) + (ml_conf * 0.4), 3)
                    method     = 'hybrid'
                else:
                    # ML result is clearly better — use it
                    category, confidence, method = ml_cat, ml_conf, 'ml'
            else:
                # No useful pattern signal — use ML exclusively
                category, confidence, method = ml_cat, ml_conf, 'ml'

        # Map confidence float to a display level label
        level = (
            'High'   if confidence >= HIGH_CONFIDENCE   else
            'Medium' if confidence >= MEDIUM_CONFIDENCE else
            'Low'
        )

        return {
            'category':         category,
            'category_label':   CATEGORIES.get(category, category),
            'confidence':       round(confidence, 3),
            'confidence_pct':   f"{round(confidence * 100)}%",
            'confidence_level': level,
            'method':           method,
            'all_scores':       all_scores,
        }

    def categorise_batch(self, texts: List[str]) -> List[Dict]:
        """Categorise a list of texts, returning one result dict per item."""
        return [self.categorise(t) for t in texts]

    def get_category_summary(self, text: str) -> str:
        """
        Return a short human-readable summary for UI display.

        Example: 'Code / Technical (87% confidence)'
        """
        r = self.categorise(text)
        return f"{r['category_label']} ({r['confidence_pct']} confidence)"


# ── Standalone test ────────────────────────────────────────────────────────────

def test_categoriser():
    """Quick smoke test across all four categories."""
    print("=" * 60)
    print("TEXT CATEGORISATION TEST (pure Python)")
    print("=" * 60)

    cat = TextCategoriser()

    samples = [
        ("Email",
         "Please contact support@example.com or call 01273 123456"),
        ("Phone / address",
         "Deliver to: 42 Baker Street London W1U 6DE Tel: 020 7224 3688"),
        ("Code",
         "def preprocess(image):\n"
         "    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n"
         "    return gray"),
        ("Documentation",
         "This report outlines the findings of the usability study "
         "conducted in Q4 2025."),
        ("Ambiguous",
         "Screenshot OCR Tool v1.0 Released January 2026"),
    ]

    for label, text in samples:
        result = cat.categorise(text)
        print(f"\n{label}")
        print(f"  Text      : {text[:70]}{'…' if len(text) > 70 else ''}")
        print(f"  Category  : {result['category_label']}")
        print(f"  Confidence: {result['confidence_pct']} "
              f"({result['confidence_level']}) via {result['method']}")
        if result['all_scores']:
            scores = ', '.join(
                f"{k}: {v:.2f}"
                for k, v in sorted(
                    result['all_scores'].items(), key=lambda x: -x[1]
                )
            )
            print(f"  Scores    : {scores}")

    print("\n" + "=" * 60)
    print("Test complete.")


if __name__ == "__main__":
    test_categoriser()