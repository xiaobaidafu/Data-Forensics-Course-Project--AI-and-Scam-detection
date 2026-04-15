"""Feature extraction utilities for suspicious text analysis."""

from __future__ import annotations

import math
import re
from collections import Counter

import nltk


def ensure_nltk_resources():
    """Download required NLTK resources if they are not already available."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK with a regex fallback."""
    ensure_nltk_resources()
    clean_text = text.strip()
    if not clean_text:
        return []

    try:
        sentences = nltk.sent_tokenize(clean_text)
    except LookupError:
        sentences = re.split(r"(?<=[.!?])\s+", clean_text)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens."""
    ensure_nltk_resources()
    clean_text = text.strip()
    if not clean_text:
        return []

    try:
        raw_tokens = nltk.word_tokenize(clean_text)
    except LookupError:
        raw_tokens = re.findall(r"\b\w+\b", clean_text)

    return [token.lower() for token in raw_tokens if re.search(r"\w", token)]


def average_sentence_length(sentences: list[str], tokens: list[str]) -> float:
    """Compute the average number of tokens per sentence."""
    if not sentences:
        return 0.0
    return len(tokens) / len(sentences)


def average_word_length(tokens: list[str]) -> float:
    """Compute the average token length."""
    if not tokens:
        return 0.0
    return sum(len(token) for token in tokens) / len(tokens)


def lexical_diversity(tokens: list[str]) -> float:
    """Measure vocabulary variety as unique words divided by total words."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def repetition_ratio(tokens: list[str]) -> float:
    """Estimate repetition by measuring repeated words relative to total words."""
    if not tokens:
        return 0.0

    counts = Counter(tokens)
    repeated_tokens = sum(count - 1 for count in counts.values() if count > 1)
    return repeated_tokens / len(tokens)


def entropy_estimate(tokens: list[str]) -> float:
    """Estimate token distribution entropy in bits."""
    if not tokens:
        return 0.0

    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0

    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


def extract_features(text: str) -> dict[str, float]:
    """Extract all numeric features used by the baseline classifier."""
    sentences = split_sentences(text)
    tokens = tokenize_text(text)

    return {
        "avg_sentence_length": average_sentence_length(sentences, tokens),
        "avg_word_length": average_word_length(tokens),
        "lexical_diversity": lexical_diversity(tokens),
        "repetition_ratio": repetition_ratio(tokens),
        "entropy": entropy_estimate(tokens),
    }
