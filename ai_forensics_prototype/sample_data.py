"""Dataset loading utilities for the phishing email prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import pandas as pd


TEXT_COLUMN_KEYWORDS = [
    "text",
    "body",
    "content",
    "email",
    "message",
    "mail",
]
LABEL_COLUMN_KEYWORDS = [
    "label",
    "class",
    "type",
    "category",
    "target",
]
POSITIVE_LABELS = {
    "1",
    "phishing",
    "phish",
    "spam",
    "fraud",
    "malicious",
    "scam",
    "unsafe",
}
NEGATIVE_LABELS = {
    "0",
    "legitimate",
    "legit",
    "safe",
    "ham",
    "benign",
    "normal",
}
DEFAULT_SAMPLE_SIZE = 800
MIN_TEXT_LENGTH = 40


@dataclass
class DatasetConfig:
    """Metadata describing the located dataset source."""

    source_type: str
    path: Path
    csv_name: str | None
    text_column: str
    label_column: str


def _score_filename(name: str) -> int:
    """Prefer CSV files with phishing- or email-related names."""
    lowered = name.lower()
    score = 0
    if "phishing" in lowered:
        score += 6
    if "email" in lowered:
        score += 3
    if "mail" in lowered:
        score += 2
    return score


def _choose_column(columns: list[str], keywords: list[str]) -> str | None:
    """Pick the best matching column based on common naming patterns."""
    lowered = {column.lower(): column for column in columns}

    for keyword in keywords:
        if keyword in lowered:
            return lowered[keyword]

    for column in columns:
        column_lower = column.lower()
        if any(keyword in column_lower for keyword in keywords):
            return column

    return None


def _inspect_csv_header(path: Path, csv_name: str | None = None) -> DatasetConfig | None:
    """Inspect a CSV header from disk or inside a zip archive."""
    if csv_name is None:
        frame = pd.read_csv(path, nrows=5)
        source_type = "csv"
    else:
        with zipfile.ZipFile(path) as archive:
            with archive.open(csv_name) as handle:
                frame = pd.read_csv(handle, nrows=5)
        source_type = "zip"

    columns = list(frame.columns)
    text_column = _choose_column(columns, TEXT_COLUMN_KEYWORDS)
    label_column = _choose_column(columns, LABEL_COLUMN_KEYWORDS)

    if text_column is None or label_column is None:
        return None

    return DatasetConfig(
        source_type=source_type,
        path=path,
        csv_name=csv_name,
        text_column=text_column,
        label_column=label_column,
    )


def detect_dataset(project_dir: str | Path | None = None) -> DatasetConfig:
    """Locate the most suitable phishing CSV in the project directory."""
    base_dir = Path(project_dir) if project_dir else Path(__file__).resolve().parent
    candidates: list[tuple[int, DatasetConfig]] = []

    for path in base_dir.iterdir():
        if path.suffix.lower() == ".csv":
            config = _inspect_csv_header(path)
            if config:
                candidates.append((_score_filename(path.name), config))
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for csv_name in archive.namelist():
                    if csv_name.lower().endswith(".csv"):
                        config = _inspect_csv_header(path, csv_name)
                        if config:
                            score = _score_filename(path.name) + _score_filename(csv_name)
                            candidates.append((score, config))

    if not candidates:
        raise FileNotFoundError(
            "No compatible phishing dataset was found in the project directory."
        )

    candidates.sort(
        key=lambda item: (item[0], item[1].csv_name or item[1].path.name),
        reverse=True,
    )
    return candidates[0][1]


def _read_dataset_frame(config: DatasetConfig) -> pd.DataFrame:
    """Read only the detected text and label columns from the located dataset."""
    columns = [config.text_column, config.label_column]

    if config.source_type == "csv":
        return pd.read_csv(config.path, usecols=columns)

    with zipfile.ZipFile(config.path) as archive:
        with archive.open(config.csv_name) as handle:
            return pd.read_csv(handle, usecols=columns)


def _normalize_label(value) -> int | None:
    """Normalize multiple label formats into binary phishing indicators."""
    if pd.isna(value):
        return None

    value_str = str(value).strip().lower()
    if value_str in POSITIVE_LABELS:
        return 1
    if value_str in NEGATIVE_LABELS:
        return 0

    try:
        numeric_value = float(value_str)
    except ValueError:
        return None

    return 1 if numeric_value > 0 else 0


def _balanced_sample(frame: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    """Keep local demo performance fast by sampling a balanced subset when possible."""
    if len(frame) <= sample_size:
        return frame.reset_index(drop=True)

    per_label = max(sample_size // 2, 1)
    sampled_parts = []

    for label in sorted(frame["label"].unique()):
        label_frame = frame[frame["label"] == label]
        take = min(len(label_frame), per_label)
        sampled_parts.append(label_frame.sample(n=take, random_state=random_state))

    sampled = pd.concat(sampled_parts)

    if len(sampled) < sample_size:
        remainder = frame.drop(sampled.index, errors="ignore")
        if not remainder.empty:
            extra = remainder.sample(
                n=min(sample_size - len(sampled), len(remainder)),
                random_state=random_state,
            )
            sampled = pd.concat([sampled, extra])

    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_training_samples(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    min_text_length: int = MIN_TEXT_LENGTH,
    random_state: int = 42,
) -> tuple[list[dict[str, str | int]], dict[str, str | int]]:
    """Load, clean, and sample phishing data for model training."""
    config = detect_dataset()
    frame = _read_dataset_frame(config).rename(
        columns={config.text_column: "text", config.label_column: "label"}
    )

    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"].str.len() >= min_text_length].copy()
    frame["label"] = frame["label"].apply(_normalize_label)
    frame = frame.dropna(subset=["label"]).copy()
    frame["label"] = frame["label"].astype(int)

    if frame["label"].nunique() < 2:
        raise ValueError("The detected dataset does not contain both phishing and legitimate labels.")

    sampled_frame = _balanced_sample(frame, sample_size=sample_size, random_state=random_state)
    samples = sampled_frame.to_dict(orient="records")
    metadata = {
        "source_path": str(config.path),
        "csv_name": config.csv_name or config.path.name,
        "text_column": config.text_column,
        "label_column": config.label_column,
        "rows_used": len(sampled_frame),
        "rows_available": len(frame),
    }
    return samples, metadata
