from torch.utils.data import Dataset


class BehavioralDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        return {
            "emotions": example["emotions"],
            "behavioral": example["behavioral"],
            "cognitive": example["cognitive"],
            "context": example["context"],
            "scenario": example["scenario"],
            "targets": example["behavioral_targets"],
        }