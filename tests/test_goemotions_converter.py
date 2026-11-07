
import json

from src.data.converters.goemotions_converter import (
    convert_goemotions_to_jsonl,
)


def test_convert_goemotions_to_jsonl(tmp_path):
    # Create a small emotion vocabulary
    emotions_file = tmp_path / "emotions.txt"
    emotions_file.write_text(
        "\n".join([
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        ]),
        encoding="utf-8",
    )

    # Create a small TSV dataset
    tsv_file = tmp_path / "sample.tsv"
    tsv_file.write_text(
        "text\tlabels\tid\n"
        "I feel nervous\t19\t001\n"
        "I am angry\t2\t002\n",
        encoding="utf-8",
    )

    output_file = tmp_path / "output" / "sample.jsonl"

    count = convert_goemotions_to_jsonl(
        tsv_file,
        emotions_file,
        output_file,
    )

    assert count == 2
    assert output_file.exists()

    records = [
        json.loads(line)
        for line in output_file.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(records) == 2

    nervous_record = records[0]

    assert len(nervous_record["emotion_vector"]) == 28
    assert nervous_record["emotion_vector"][19] == 1.0
    assert nervous_record["project_emotions"]["anxiety"] == 100.0
    assert nervous_record["project_emotion_mask"]["anxiety"] == 1.0

    angry_record = records[1]

    assert angry_record["emotion_vector"][2] == 1.0
    assert angry_record["project_emotion_mask"]["anger"] == 1.0
