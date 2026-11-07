
import json
from pathlib import Path

from src.data.adapters.goemotions import (
    load_emotion_names,
    parse_goemotions_tsv,
)

from src.data.mappings.goemotions_to_project import (
    map_goemotions_to_project,
    get_project_emotion_mask,
)


def convert_goemotions_to_jsonl(
    tsv_path,
    emotions_path,
    output_path,
):
    """
    Convert GoEmotions TSV data into processed JSONL records.
    """

    tsv_path = Path(tsv_path)
    emotions_path = Path(emotions_path)
    output_path = Path(output_path)

    # Load the original 28 emotion names
    emotion_names = load_emotion_names(emotions_path)

    # Parse the original TSV records
    records = parse_goemotions_tsv(
        tsv_path,
        emotions_path,
    )

    # Create output directory if necessary
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    converted_count = 0

    with output_path.open("w", encoding="utf-8") as output_file:

        for record in records:

            # Create a 28-dimensional multi-hot vector
            emotion_vector = [0.0] * len(emotion_names)

            for emotion_id in record["source_emotion_ids"]:
                if emotion_id not in emotion_names:
                    raise ValueError(
                        f"Unknown emotion ID: {emotion_id}"
                    )

                emotion_vector[emotion_id] = 1.0

            # Create dictionary representation
            emotion_vector_dict = {
                emotion_names[index]: emotion_vector[index]
                for index in range(len(emotion_names))
            }

            # Map source labels to project emotions
            project_emotions = map_goemotions_to_project(
                record["source_emotions"]
            )

            # Identify which project dimensions have supervision
            project_emotion_mask = get_project_emotion_mask(
                record["source_emotions"]
            )

            # Build final record
            converted_record = {
                **record,
                "emotion_vector": emotion_vector,
                "emotion_vector_dict": emotion_vector_dict,
                "project_emotions": project_emotions,
                "project_emotion_mask": project_emotion_mask,
            }

            output_file.write(
                json.dumps(converted_record, ensure_ascii=False)
                + "\n"
            )

            converted_count += 1

    return converted_count
