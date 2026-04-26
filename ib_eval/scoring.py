"""Answer extraction and exact-match scoring."""

import re

_TOKEN_RE = re.compile(r"[A-Za-z]+")
# Common filler words models prepend — never the answer.
_STOPWORDS = {
    "the", "a", "an", "answer", "is", "person", "paired", "with", "lives",
    "in", "to", "of", "for", "based", "given", "according", "from", "they",
    "he", "she", "it", "would", "be", "would", "live",
}


def extract_answer(
    raw: str,
    vocab: set[str] | None = None,
    exclude: set[str] | None = None,
) -> str:
    """Pull the most likely answer name from the model's reply.

    Strategy:
      1. Tokenize alphabetic words.
      2. Drop any token whose lowercase form is in `exclude` (e.g. the question
         subject — it's never the answer).
      3. If `vocab` is given, return the LAST vocab-matching token (verbose
         replies tend to put the answer at the end of the sentence).
      4. Otherwise return the first non-stopword token.
    """
    text = raw.strip().strip(".,!?\"'")
    tokens = _TOKEN_RE.findall(text)
    excl = {x.lower() for x in (exclude or set())}
    tokens = [t for t in tokens if t.lower() not in excl]

    if vocab:
        vocab_hits = [t for t in tokens if t.lower() in vocab]
        if vocab_hits:
            return vocab_hits[-1]
    for tok in tokens:
        if tok.lower() not in _STOPWORDS:
            return tok
    return tokens[0] if tokens else ""


def score(prediction: str, ground_truth: str) -> bool:
    return prediction.strip().lower() == ground_truth.strip().lower()


def build_vocab(dataset: list[dict]) -> set[str]:
    vocab: set[str] = set()
    for ex in dataset:
        for a, b in ex["facts"]:
            vocab.add(a.lower())
            vocab.add(b.lower())
        for city in ex["attributes"].values():
            vocab.add(city.lower())
    return vocab
