from src.data.jsonl_loader import load_jsonl
from src.data.loader import create_dataloader


def test_dataloader():

    examples = load_jsonl(
        "data/raw/behavioral_data.jsonl"
    )

    loader = create_dataloader(
        examples,
        batch_size=2,
    )

    batch = next(iter(loader))

    assert batch["profile"].shape == (2, 27)
    assert batch["targets"].shape == (2, 6)
    assert len(batch["scenario"]) == 2