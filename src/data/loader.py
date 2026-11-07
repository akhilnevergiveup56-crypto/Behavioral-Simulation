
from torch.utils.data import DataLoader

from src.data.dataset import BehavioralDataset
from src.data.collate import collate_examples
from src.data.goemotions_dataset import GoEmotionsDataset


def create_dataloader(
    examples: list[dict],
    batch_size: int = 2,
    shuffle: bool = False,
):
    """
    Create a DataLoader for the existing behavioral dataset.
    """

    dataset = BehavioralDataset(examples)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_examples,
    )


def create_goemotions_dataloader(
    data,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    shuffle: bool = False,
):
    """
    Create a DataLoader for the GoEmotions dataset.
    """

    dataset = GoEmotionsDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
