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
    """Pull the most likely answer name from a free-text response."""
    text = raw.strip().strip(".,!?\"'")
    tokens = _TOKEN_RE.findall(text)
    excluded = {item.lower() for item in (exclude or set())}
    tokens = [token for token in tokens if token.lower() not in excluded]

    if vocab:
        vocab_hits = [token for token in tokens if token.lower() in vocab]
        if vocab_hits:
            return vocab_hits[-1]
    for token in tokens:
        if token.lower() not in _STOPWORDS:
            return token
    return tokens[0] if tokens else ""


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
