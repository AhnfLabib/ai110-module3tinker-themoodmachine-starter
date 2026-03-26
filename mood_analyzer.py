# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
import string
from typing import List, Optional, Tuple

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Strip from token ends after lowercasing; includes common Unicode quotes.
# Omit "_" — it is in string.punctuation but placeholders use "__emoticon_pos__", etc.
_EXTRA_STRIP_CHARS = "\u2018\u2019\u201c\u201d…"
_STRIP_CHARS = string.punctuation.replace("_", "") + _EXTRA_STRIP_CHARS

# Applied longest-first so ":-)" does not leave a stray "(" behind.
_MOOD_SUBSTITUTIONS: Tuple[Tuple[str, str], ...] = tuple(
    sorted(
        (
            (":-(", "__emoticon_neg__"),
            (":-)", "__emoticon_pos__"),
            (":(", "__emoticon_neg__"),
            (":)", "__emoticon_pos__"),
            ("\U0001f602", "__emoji_pos__"),  # 😂
            ("\U0001f642", "__emoji_pos__"),  # 🙂
            ("\U0001f60a", "__emoji_pos__"),  # 😊
            ("\U0001f972", "__emoji_neg__"),  # 🥲
            ("\U0001f480", "__emoji_neg__"),  # 💀
            ("\U0001f622", "__emoji_neg__"),  # 😢
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
)


def _apply_mood_substitutions(text: str) -> str:
    out = text
    for needle, repl in _MOOD_SUBSTITUTIONS:
        out = out.replace(needle, f" {repl} ")
    return out


def _strip_outer_punctuation(token: str) -> str:
    return token.strip(_STRIP_CHARS)


def _collapse_repeated_chars(token: str) -> str:
    # Collapse 3+ identical characters to 2 (e.g. "soooo" -> "soo"); letters only.
    if not token.isalpha():
        return token
    return re.sub(r"(.)\1{2,}", r"\1\1", token)


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        - Lowercase, trim, map ASCII faces and selected emoji to lexicon placeholders
        - Split on whitespace
        - Strip leading/trailing punctuation (ASCII + common Unicode quotes)
        - Collapse 3+ repeated letters to 2 (e.g. "soooo" -> "soo")
        - Drop empty tokens
        """
        cleaned = _apply_mood_substitutions(text.strip().lower())
        raw_tokens = cleaned.split()
        tokens: List[str] = []
        for tok in raw_tokens:
            t = _strip_outer_punctuation(tok)
            t = _collapse_repeated_chars(t)
            if t:
                tokens.append(t)

        print(tokens)
        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        tokens = self.preprocess(text)
        score = 0
        for token in tokens:
            if token in self.positive_words:
                score += 1
            if token in self.negative_words:
                score -= 1
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        Mapping from ``score_text``:
          - score >= 2  -> "positive"
          - score <= -2 -> "negative"
          - score == 0  -> "neutral"
          - otherwise   -> "mixed" (e.g. scores 1 or -1)

        Labels should align with ``TRUE_LABELS`` in dataset.py when measuring accuracy.
        """
        score = self.score_text(text)
        if score >= 2:
            return "positive"
        if score <= -2:
            return "negative"
        if score == 0:
            return "neutral"
        return "mixed"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
