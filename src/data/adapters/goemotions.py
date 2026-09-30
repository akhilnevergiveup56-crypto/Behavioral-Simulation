from pathlib import Path


def inspect_goemotions_file(path: str, n: int = 5):
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i >= n:
                break

            parts = line.rstrip("\n").split("\t")

            print(f"Row {i + 1}")
            print(f"Number of fields: {len(parts)}")
            print(f"Fields: {parts}")
            print("-" * 50)