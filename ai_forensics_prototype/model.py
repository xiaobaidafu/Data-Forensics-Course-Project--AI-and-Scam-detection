"""Baseline model utilities for classifying suspicious text."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extraction import extract_features


FEATURE_COLUMNS = [
    "avg_sentence_length",
    "avg_word_length",
    "lexical_diversity",
    "repetition_ratio",
    "entropy",
]
DATASET_PATH = Path(__file__).resolve().parent / "ai_cybercrime_dataset.py"
MAX_TRAINING_SAMPLES = 1000
MIN_TEXT_LENGTH = 25
TEXT_COLUMN_CANDIDATES = ["text", "body", "content", "email", "message", "input"]
LABEL_COLUMN_CANDIDATES = ["label", "class", "type", "target", "y"]


def _extract_raw_dataset():
    """Parse the dataset Python file into a native Python object."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    raw_text = DATASET_PATH.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError("The dataset file is empty.")

    try:
        return ast.literal_eval(raw_text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            "Could not parse ai_cybercrime_dataset.py as a Python literal dataset."
        ) from exc


def _pick_matching_column(columns: list[str], candidates: list[str]) -> str | None:
    """Choose the best matching column from a dataframe-like dataset."""
    lowered = {column.lower(): column for column in columns}

    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    for column in columns:
        if any(candidate in column.lower() for candidate in candidates):
            return column

    return None


def _coerce_to_dataset_frame(raw_dataset) -> tuple[pd.DataFrame, dict[str, str]]:
    """Convert common Python dataset structures into a text/label dataframe."""
    metadata = {"dataset_file": str(DATASET_PATH.name), "dataset_format": type(raw_dataset).__name__}

    if isinstance(raw_dataset, pd.DataFrame):
        frame = raw_dataset.copy()
    elif isinstance(raw_dataset, list):
        if not raw_dataset:
            raise ValueError("The dataset file does not contain any samples.")

        first_item = raw_dataset[0]
        if isinstance(first_item, dict):
            frame = pd.DataFrame(raw_dataset)
            text_column = _pick_matching_column(list(frame.columns), TEXT_COLUMN_CANDIDATES)
            label_column = _pick_matching_column(list(frame.columns), LABEL_COLUMN_CANDIDATES)
            if text_column is None or label_column is None:
                raise ValueError("Could not detect text and label columns in the dataset dictionaries.")
            frame = frame.rename(columns={text_column: "text", label_column: "label"})
            metadata["text_source_column"] = text_column
            metadata["label_source_column"] = label_column
        elif isinstance(first_item, (list, tuple)) and len(first_item) >= 2:
            frame = pd.DataFrame(raw_dataset, columns=["text", "label"])
        else:
            raise ValueError("Unsupported list dataset format. Expected tuples/lists or dictionaries.")
    else:
        try:
            frame = pd.DataFrame(raw_dataset)
        except Exception as exc:
            raise ValueError("Unsupported dataset structure in ai_cybercrime_dataset.py.") from exc

        text_column = _pick_matching_column(list(frame.columns), TEXT_COLUMN_CANDIDATES)
        label_column = _pick_matching_column(list(frame.columns), LABEL_COLUMN_CANDIDATES)
        if text_column is None or label_column is None:
            raise ValueError("Could not detect text and label columns in the dataset structure.")
        frame = frame.rename(columns={text_column: "text", label_column: "label"})
        metadata["text_source_column"] = text_column
        metadata["label_source_column"] = label_column

    if "text" not in frame.columns or "label" not in frame.columns:
        raise ValueError("The dataset could not be normalized into text and label columns.")

    return frame[["text", "label"]].copy(), metadata


def _normalize_label(value) -> int | None:
    """Normalize common label values into binary integers."""
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return int(value)

    value_str = str(value).strip().lower()
    if value_str in {"1", "phishing", "ai", "generated", "machine", "spam", "fraud", "malicious"}:
        return 1
    if value_str in {"0", "human", "legitimate", "safe", "benign", "ham", "real"}:
        return 0

    try:
        numeric_value = float(value_str)
    except ValueError:
        return None

    if numeric_value in {0.0, 1.0}:
        return int(numeric_value)

    return None


def load_training_samples(max_samples: int = MAX_TRAINING_SAMPLES) -> tuple[list[dict[str, str | int]], dict]:
    """Load, validate, clean, and sample the new Python-based dataset."""
    raw_dataset = _extract_raw_dataset()
    frame, metadata = _coerce_to_dataset_frame(raw_dataset)

    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"].str.len() >= MIN_TEXT_LENGTH].copy()
    frame["label"] = frame["label"].apply(_normalize_label)
    frame = frame.dropna(subset=["label"]).copy()
    frame["label"] = frame["label"].astype(int)

    if frame.empty:
        raise ValueError("No valid rows remained after dataset cleaning.")
    if frame["label"].nunique() < 2:
        raise ValueError("The dataset must contain at least two label classes for training.")

    if len(frame) > max_samples:
        sampled_parts = []
        per_class = max(max_samples // frame["label"].nunique(), 1)
        for label_value, group in frame.groupby("label"):
            sampled_parts.append(group.sample(n=min(len(group), per_class), random_state=42))
        frame = pd.concat(sampled_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

    label_distribution = Counter(frame["label"])
    metadata["samples_loaded"] = int(len(frame))
    metadata["label_distribution"] = dict(sorted(label_distribution.items()))
    return frame.to_dict(orient="records"), metadata


def build_training_frame() -> pd.DataFrame:
    """Build a feature dataframe from the new Python dataset."""
    samples, metadata = load_training_samples()
    rows = []

    for sample in samples:
        features = extract_features(sample["text"])
        features["label"] = sample["label"]
        rows.append(features)

    training_frame = pd.DataFrame(rows)
    training_frame.attrs["dataset_metadata"] = metadata
    return training_frame


def train_baseline_model() -> dict:
    """Train a lightweight logistic regression classifier."""
    training_frame = build_training_frame()
    x_train = training_frame[FEATURE_COLUMNS]
    y_train = training_frame["label"]

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(random_state=42)),
        ]
    )
    classifier.fit(x_train, y_train)

    dataset_metadata = training_frame.attrs.get("dataset_metadata", {})
    print(
        "Training summary:",
        {
            "samples_loaded": dataset_metadata.get("samples_loaded", len(training_frame)),
            "label_distribution": dataset_metadata.get(
                "label_distribution",
                training_frame["label"].value_counts().sort_index().to_dict(),
            ),
            "training_succeeded": True,
        },
    )

    return {
        "model": classifier,
        "training_frame": training_frame,
        "dataset_metadata": dataset_metadata,
    }


def predict_text(text: str, model_bundle: dict) -> dict[str, float | str]:
    """Predict whether the input text is AI-generated or human-written.

    The prototype blends the learned model probability with a lightweight
    feature heuristic so the demo remains stable despite the tiny dataset.
    """
    features = extract_features(text)
    feature_frame = pd.DataFrame([features])[FEATURE_COLUMNS]
    probabilities = model_bundle["model"].predict_proba(feature_frame)[0]
    class_labels = list(model_bundle["model"].classes_)

    probability_map = dict(zip(class_labels, probabilities))
    model_ai_probability = float(probability_map.get(1, 0.0))

    heuristic_ai_probability = 0.0
    heuristic_ai_probability += min(features["avg_sentence_length"] / 20.0, 1.0) * 0.25
    heuristic_ai_probability += min(features["avg_word_length"] / 6.0, 1.0) * 0.15
    heuristic_ai_probability += (1.0 - min(features["lexical_diversity"], 1.0)) * 0.20
    heuristic_ai_probability += min(features["repetition_ratio"] / 0.35, 1.0) * 0.20
    heuristic_ai_probability += min(features["entropy"] / 5.0, 1.0) * 0.20

    ai_probability = (0.7 * model_ai_probability) + (0.3 * heuristic_ai_probability)

    label = "Likely AI-generated" if ai_probability >= 0.5 else "Likely Human-written"
    confidence = ai_probability if ai_probability >= 0.5 else 1.0 - ai_probability

    return {
        "label": label,
        "confidence": confidence,
        "ai_probability": ai_probability,
        "model_probability": model_ai_probability,
    }


def explain_prediction(features: dict[str, float], label: str) -> str:
    """Generate a natural presentation-ready explanation from extracted features."""
    observations = []

    if features["avg_sentence_length"] >= 16:
        observations.append("longer and more structured sentences")
    elif features["avg_sentence_length"] <= 10:
        observations.append("shorter sentence construction")

    if features["repetition_ratio"] >= 0.22:
        observations.append("noticeable repetition across key terms")

    if features["lexical_diversity"] <= 0.62:
        observations.append("more limited vocabulary variation")
    elif features["lexical_diversity"] >= 0.72:
        observations.append("stronger vocabulary variety")

    if features["entropy"] <= 3.5:
        observations.append("lower overall token variation")
    elif features["entropy"] >= 4.0:
        observations.append("broader token variation")

    if not observations:
        observations.append("a balanced mix of linguistic signals")

    lead = ", ".join(observations[:3])

    if label == "Likely AI-generated":
        return (
            f"The writing shows {lead}. In this prototype, that pattern is more "
            "consistent with AI-assisted or highly templated cybercrime content."
        )

    return (
        f"The writing shows {lead}. In this prototype, that profile leans more "
        "toward human-written suspicious communication than machine-generated text."
    )
