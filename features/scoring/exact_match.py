"""Answer extraction and exact-match scoring."""

import re

_TOKEN_RE = re.compile(r"[A-Za-z]+")
_STOPWORDS = {
    "the", "a", "an", "answer", "is", "person", "paired", "with", "lives",
    "in", "to", "of", "for", "based", "given", "according", "from", "they",
    "he", "she", "it", "would", "be", "live",
}


def extract_answer(
    raw: str,
    vocab: set[str] | None = None,
    exclude: set[str] | None = None,
) -> str:
    """Pull the most likely answer name from a free-text response.

    The `exclude` set (e.g. the question subject) is used to prefer a
    *better* token, but if no non-excluded token is found we fall back to
    the raw first vocab token. This ensures we never return "" when the
    model did produce output — an empty string would be silently counted as
    wrong without recording what the model actually said.
    """
    text = raw.strip().strip(".,!?\"'")
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return ""

    excluded = {item.lower() for item in (exclude or set())}
    preferred = [t for t in tokens if t.lower() not in excluded]

    # Try preferred tokens first (exclude the question subject)
    if vocab:
        vocab_hits = [t for t in preferred if t.lower() in vocab]
        if vocab_hits:
            return vocab_hits[-1]
    for token in preferred:
        if token.lower() not in _STOPWORDS:
            return token

    # Fallback: no preferred token found — return best token ignoring exclude
    # so the prediction is recorded (likely wrong, but not silently empty)
    if vocab:
        vocab_hits = [t for t in tokens if t.lower() in vocab]
        if vocab_hits:
            return vocab_hits[-1]
    for token in tokens:
        if token.lower() not in _STOPWORDS:
            return token
    return tokens[0]


def score(prediction: str, ground_truth: str) -> bool:
    """Strict normalized exact match used as the primary metric."""
    return prediction.strip().lower() == ground_truth.strip().lower()


def build_vocab(dataset: list[dict]) -> set[str]:
    """Build the response vocabulary used by answer extraction."""
    vocab: set[str] = set()
    for example in dataset:
        for left_name, right_name in example["facts"]:
            vocab.add(left_name.lower())
            vocab.add(right_name.lower())
        for city in example["attributes"].values():
            vocab.add(city.lower())
    return vocab
