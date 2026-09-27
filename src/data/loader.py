from torch.utils.data import DataLoader

from src.data.dataset import BehavioralDataset
from src.data.collate import collate_examples


def create_dataloader(
    examples: list[dict],
    batch_size: int = 2,
    shuffle: bool = False,
):
    dataset = BehavioralDataset(examples)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_examples,
    )