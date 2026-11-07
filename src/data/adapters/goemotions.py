
from pathlib import Path


def inspect_goemotions_file(path: str, n: int = 5):
    """Print the first n rows of a GoEmotions TSV file."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i >= n:
                break

            parts = line.rstrip("\r\n").split("\t")

            print(f"Row {i + 1}")
            print(f"Number of fields: {len(parts)}")
            print(f"Fields: {parts}")
            print("-" * 50)


def load_emotion_names(path):
    """Load emotion names and map each name to its integer ID."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Emotion file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        names = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if not names:
        raise ValueError("Emotion vocabulary is empty.")

    return {
        index: name
        for index, name in enumerate(names)
    }


def parse_label_ids(label_field):
    """Convert comma-separated label IDs into integers."""

    if not isinstance(label_field, str) or not label_field.strip():
        raise ValueError("Label field cannot be empty.")

    label_ids = []

    for value in label_field.split(","):
        value = value.strip()

        try:
            label_id = int(value)
        except ValueError:
            raise ValueError(
                f"Invalid emotion label ID: {value}"
            )

        if label_id not in label_ids:
            label_ids.append(label_id)

    return label_ids


def parse_goemotions_tsv(tsv_path, emotions_path):
    """
    Parse a GoEmotions TSV file into structured records.

    Supports files both with and without a header row.
    """

    tsv_path = Path(tsv_path)
    emotion_names = load_emotion_names(emotions_path)

    if not tsv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {tsv_path}"
        )

    records = []

    with tsv_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.rstrip("\r\n")

            if not line:
                continue

            parts = line.split("\t", maxsplit=2)

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid TSV format at line {line_number}: "
                    f"expected 3 columns, received {len(parts)}."
                )

            # Detect and skip an optional header row.
            if line_number == 1:
                header = [
                    part.strip().lower()
                    for part in parts
                ]

                if (
                    header[0] == "text"
                    and header[1] in {"label", "labels"}
                    and header[2] in {"id", "example_id"}
                ):
                    continue

            text, label_field, example_id = parts

            source_emotion_ids = parse_label_ids(label_field)

            # Check whether all label IDs exist.
            unknown_ids = [
                emotion_id
                for emotion_id in source_emotion_ids
                if emotion_id not in emotion_names
            ]

            if unknown_ids:
                raise ValueError(
                    f"Unknown emotion IDs {unknown_ids} "
                    f"at line {line_number}."
                )

            # Convert integer IDs into emotion names.
            source_emotions = [
                emotion_names[emotion_id]
                for emotion_id in source_emotion_ids
            ]

            records.append({
                "source": "goemotions",
                "source_id": example_id,
                "text": text,
                "source_emotion_ids": source_emotion_ids,
                "source_emotions": source_emotions,
            })

    return records
