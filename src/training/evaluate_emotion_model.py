
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from src.data.loader import create_goemotions_dataloader
from src.data.adapters.goemotions import load_emotion_names
from src.models.emotion_encoder import MultiTaskEmotionEncoder


def get_device():
    """Choose the best available computing device."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def calculate_metrics(tp, fp, fn):
    """Calculate micro and macro classification metrics."""

    tp = tp.float()
    fp = fp.float()
    fn = fn.float()

    support = tp + fn

    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / support.clamp_min(1.0)

    f1 = (
        2 * precision * recall
        / (precision + recall).clamp_min(1e-12)
    )

    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()

    micro_precision = (
        total_tp / (total_tp + total_fp).clamp_min(1.0)
    )

    micro_recall = (
        total_tp / (total_tp + total_fn).clamp_min(1.0)
    )

    micro_f1 = (
        2 * micro_precision * micro_recall
        / (micro_precision + micro_recall).clamp_min(1e-12)
    )

    # Average only labels that actually appear in this evaluation set.
    supported = support > 0

    if supported.any():
        macro_precision = precision[supported].mean()
        macro_recall = recall[supported].mean()
        macro_f1 = f1[supported].mean()
    else:
        macro_precision = torch.tensor(0.0)
        macro_recall = torch.tensor(0.0)
        macro_f1 = torch.tensor(0.0)

    return {
        "micro_precision": micro_precision.item(),
        "micro_recall": micro_recall.item(),
        "micro_f1": micro_f1.item(),
        "macro_precision": macro_precision.item(),
        "macro_recall": macro_recall.item(),
        "macro_f1": macro_f1.item(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
    }


def evaluate_model(model, dataloader, device, threshold=0.5):
    """Evaluate the 28-label GoEmotions prediction head."""

    model.eval()

    tp = torch.zeros(28)
    fp = torch.zeros(28)
    fn = torch.zeros(28)

    example_count = 0

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            targets = batch["goemotions_targets"].cpu()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probabilities = torch.sigmoid(
                outputs["goemotions_logits"]
            )

            predictions = (
                probabilities >= threshold
            ).cpu().float()

            tp += (predictions * targets).sum(dim=0)
            fp += (predictions * (1 - targets)).sum(dim=0)
            fn += ((1 - predictions) * targets).sum(dim=0)

            example_count += input_ids.size(0)

    if example_count == 0:
        raise ValueError("No evaluation examples were processed.")

    metrics = calculate_metrics(tp, fp, fn)
    metrics["examples_evaluated"] = example_count

    return metrics


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate the trained GoEmotions model."
    )

    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
    )

    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("Batch size must be positive.")

    if not 0 < args.threshold < 1:
        parser.error("Threshold must be between 0 and 1.")

    if args.max_samples is not None and args.max_samples < 1:
        parser.error("Maximum samples must be positive.")

    project_root = Path(__file__).resolve().parents[2]

    checkpoint_dir = (
        project_root / "checkpoints" / "goemotions-distilbert"
    )

    checkpoint_path = checkpoint_dir / "best_model.pt"
    tokenizer_path = checkpoint_dir / "tokenizer"

    data_path = (
        project_root
        / "data"
        / "processed"
        / "goemotions"
        / f"{args.split}.jsonl"
    )

    emotions_path = (
        project_root
        / "data"
        / "external"
        / "goemotions"
        / "emotions.txt"
    )

    for path in [
        checkpoint_path,
        tokenizer_path,
        data_path,
        emotions_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    device = get_device()

    print("Evaluation device:", device)
    print("Evaluation split:", args.split)

    # Load the saved checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model_name = checkpoint["model_name"]

    # Load the tokenizer saved during training.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Recreate the model architecture.
    model = MultiTaskEmotionEncoder(
        model_name=model_name,
        num_goemotions=28,
        num_project_emotions=8,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Load validation data.
    dataloader = create_goemotions_dataloader(
        data=data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=checkpoint["max_length"],
        shuffle=False,
    )

    # Limit examples for a quick evaluation.
    if args.max_samples is not None:

        sample_count = min(
            args.max_samples,
            len(dataloader.dataset),
        )

        subset = Subset(
            dataloader.dataset,
            range(sample_count),
        )

        dataloader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
        )

    emotion_names = load_emotion_names(emotions_path)

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Examples to evaluate:", len(dataloader.dataset))
    print("Prediction threshold:", args.threshold)

    # Evaluate model predictions.
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=args.threshold,
    )

    print("\n========== EVALUATION RESULTS ==========")

    print("\nMicro metrics:")
    print(f"Precision: {metrics['micro_precision']:.4f}")
    print(f"Recall:    {metrics['micro_recall']:.4f}")
    print(f"F1-score:  {metrics['micro_f1']:.4f}")

    print("\nMacro metrics (labels with validation support):")
    print(f"Precision: {metrics['macro_precision']:.4f}")
    print(f"Recall:    {metrics['macro_recall']:.4f}")
    print(f"F1-score:  {metrics['macro_f1']:.4f}")

    print("\nPer-emotion results:")

    for index in range(28):

        name = emotion_names[index]

        print(
            f"{name:15} "
            f"Precision: {metrics['per_class_precision'][index]:.4f} "
            f"Recall: {metrics['per_class_recall'][index]:.4f} "
            f"F1: {metrics['per_class_f1'][index]:.4f} "
            f"Support: {int(metrics['support'][index])}"
        )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
