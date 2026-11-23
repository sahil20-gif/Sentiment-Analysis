import re
import html

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
SPACE_RE = re.compile(r"\s+")

def clean_for_classical(text: str) -> str:
    """Light cleaning: preserve negation, punctuation cues, and word order."""
    text = html.unescape(str(text))
    text = URL_RE.sub(" URL ", text)
    text = text.replace("\x00", " ")
    text = SPACE_RE.sub(" ", text).strip()
    return text.lower()

def clean_for_sequence(text: str) -> str:
    # Keep apostrophes so contractions such as don't remain interpretable.
    text = clean_for_classical(text)
    text = re.sub(r"[^a-z0-9'!?., ]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()
