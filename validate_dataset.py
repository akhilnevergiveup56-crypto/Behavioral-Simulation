from src.data.jsonl_loader import load_jsonl
from src.data.validator import validate_dataset


DATA_PATH = "data/raw/behavioral_data.jsonl"


def main():
    examples = load_jsonl(DATA_PATH)

    validate_dataset(examples)

    print(
        f"Dataset validation successful: "
        f"{len(examples)} examples checked."
    )


if __name__ == "__main__":
    main()