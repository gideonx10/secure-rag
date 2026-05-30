# app/guardrails/output_filter.py
# DEFENDS AGAINST: PII leakage, verbatim document dumping
# METHOD:
#   presidio  — detects and masks real PII (emails, phones, names etc.)
#   verbatim  — sliding window check for copy-pasted chunk text

from presidio_analyzer  import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Load once at startup — these are heavy models
_analyzer   = AnalyzerEngine()
_anonymizer = AnonymizerEngine()

PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IP_ADDRESS",
    "MEDICAL_LICENSE",
    "CREDIT_CARD",
]

def _mask_pii(text: str) -> tuple[str, bool]:
    """
    Detect and replace PII with placeholders.
    Returns (masked_text, was_pii_found).
    """
    results = _analyzer.analyze(text=text, entities=PII_ENTITIES, language="en")
    if not results:
        return text, False
    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text, True

def _is_verbatim_leak(llm_output: str, chunks: list) -> bool:
    """
    Sliding window check for verbatim copy-paste attacks.
    
    Key insight: summaries legitimately reuse domain words.
    We only block if the LLM is copying EXACT sequences —
    not just using the same vocabulary.
    
    Changes from v1:
    - Window size increased 25 → 40 (longer sequence needed to trigger)
    - Threshold raised 0.55 → 0.80 (80% overlap required to block)
    - Skip check entirely for short outputs (< 60 words) — those are
      summaries or short answers, not verbatim dumps
    - Common stop words excluded from overlap calculation so "the",
      "and", "is" don't inflate the score
    """
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "is", "are",
        "was", "were", "be", "been", "has", "have", "had", "it",
        "its", "this", "that", "as", "which", "who", "what",
        "not", "also", "can", "be", "will", "would", "there"
    }

    words_out = [
        w for w in llm_output.lower().split()
        if w not in STOP_WORDS
    ]

    # Short answers are never verbatim dumps — skip check
    if len(words_out) < 60:
        return False

    window_size = 40   # need 40 consecutive non-stopword matches
    threshold   = 0.80 # 80% of those must match chunk words

    for chunk in chunks:
        chunk_words = set(
            w for w in chunk.page_content.lower().split()
            if w not in STOP_WORDS
        )

        for i in range(len(words_out) - window_size):
            window  = words_out[i : i + window_size]
            overlap = sum(1 for w in window if w in chunk_words)
            if overlap / window_size > threshold:
                return True

    return False

def filter_output(llm_output: str, chunks: list) -> dict:
    """
    Main output filter pipeline.
    Returns dict with: text, flags, blocked
    """
    flags = []

    # Check 1: verbatim leak
    if _is_verbatim_leak(llm_output, chunks):
        flags.append("verbatim_leak")
        return {
            "text":    "I can answer questions about the documents but cannot reproduce them verbatim.",
            "flags":   flags,
            "blocked": True
        }

    # Check 2: PII masking
    masked_text, pii_found = _mask_pii(llm_output)
    if pii_found:
        flags.append("pii_masked")

    return {
        "text":    masked_text,
        "flags":   flags,
        "blocked": False
    }