
from pathlib import Path

from src.data.converters.goemotions_converter import (
    convert_goemotions_to_jsonl,
)


def main():
    # Project root directory
    project_root = Path(__file__).resolve().parent

    # Original GoEmotions dataset location
    raw_dir = project_root / "data" / "external" / "goemotions"

    # Processed dataset destination
    output_dir = (
        project_root / "data" / "processed" / "goemotions"
    )

    emotions_file = raw_dir / "emotions.txt"

    # Check that the original emotion vocabulary exists
    if not emotions_file.exists():
        raise FileNotFoundError(
            f"Missing emotion vocabulary: {emotions_file}"
        )

    # Convert all three dataset splits
    splits = ["train", "dev", "test"]

    for split in splits:
        input_file = raw_dir / f"{split}.tsv"
        output_file = output_dir / f"{split}.jsonl"

        if not input_file.exists():
            raise FileNotFoundError(
                f"Missing dataset split: {input_file}"
            )

        print(f"\nConverting {split} dataset...")

        count = convert_goemotions_to_jsonl(
            input_file,
            emotions_file,
            output_file,
        )

        print(f"Converted records: {count}")
        print(f"Saved to: {output_file}")

    print("\nGoEmotions conversion completed!")


if __name__ == "__main__":
    main()
