
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from src.data.loader import create_goemotions_dataloader
from src.models.emotion_encoder import (
    MultiTaskEmotionEncoder,
    compute_multitask_loss,
)


def get_device():
    """Select the best available training device."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")

    return torch.device("cpu")


def limit_dataloader(loader, max_samples=None, shuffle=False):
    """Optionally limit dataset size for a small test run."""

    if max_samples is None:
        return loader

    sample_count = min(max_samples, len(loader.dataset))

    subset = Subset(
        loader.dataset,
        range(sample_count),
    )

    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
    )


def run_epoch(
    model,
    dataloader,
    device,
    optimizer=None,
):
    """
    Run one training or validation epoch.
    """

    training = optimizer is not None
    model.train(training)

    total_loss_sum = 0.0
    main_loss_sum = 0.0
    auxiliary_loss_sum = 0.0
    example_count = 0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        goemotions_targets = batch[
            "goemotions_targets"
        ].to(device)

        project_targets = batch[
            "project_emotion_targets"
        ].to(device)

        project_mask = batch[
            "project_emotion_mask"
        ].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            losses = compute_multitask_loss(
                outputs=outputs,
                goemotions_targets=goemotions_targets,
                project_emotion_targets=project_targets,
                project_emotion_mask=project_mask,
                auxiliary_weight=0.3,
            )

            loss = losses["total_loss"]

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        batch_size = input_ids.size(0)

        total_loss_sum += loss.item() * batch_size

        main_loss_sum += (
            losses["main_loss"].item() * batch_size
        )

        auxiliary_loss_sum += (
            losses["auxiliary_loss"].item() * batch_size
        )

        example_count += batch_size

    if example_count == 0:
        raise ValueError("DataLoader produced no examples.")

    return {
        "total_loss": total_loss_sum / example_count,
        "main_loss": main_loss_sum / example_count,
        "auxiliary_loss": auxiliary_loss_sum / example_count,
    }


def main():

    parser = argparse.ArgumentParser(
        description="Train the GoEmotions emotion encoder."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive.")

    if args.learning_rate <= 0 or args.max_length < 1:
        parser.error("Learning rate and max length must be positive.")

    project_root = Path(__file__).resolve().parents[2]

    train_path = (
        project_root
        / "data"
        / "processed"
        / "goemotions"
        / "train.jsonl"
    )

    val_path = (
        project_root
        / "data"
        / "processed"
        / "goemotions"
        / "dev.jsonl"
    )

    output_dir = (
        project_root
        / "checkpoints"
        / "goemotions-distilbert"
    )

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    if not val_path.exists():
        raise FileNotFoundError(val_path)

    device = get_device()

    print("Training device:", device)
    print("Loading tokenizer...")

    model_name = "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Creating DataLoaders...")

    train_loader = create_goemotions_dataloader(
        data=train_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
    )

    val_loader = create_goemotions_dataloader(
        data=val_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
    )

    train_loader = limit_dataloader(
        train_loader,
        args.max_train_samples,
        shuffle=True,
    )

    val_loader = limit_dataloader(
        val_loader,
        args.max_val_samples,
        shuffle=False,
    )

    print("Training examples:", len(train_loader.dataset))
    print("Validation examples:", len(val_loader.dataset))

    print("Loading emotion model...")

    model = MultiTaskEmotionEncoder(
        model_name=model_name,
        num_goemotions=28,
        num_project_emotions=8,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    best_val_loss = float("inf")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(args.epochs):

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
        )

        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            optimizer=None,
        )

        print("\nTraining:")
        print(f"Total loss: {train_metrics['total_loss']:.4f}")
        print(f"Main loss: {train_metrics['main_loss']:.4f}")
        print(
            f"Auxiliary loss: "
            f"{train_metrics['auxiliary_loss']:.4f}"
        )

        print("\nValidation:")
        print(f"Total loss: {val_metrics['total_loss']:.4f}")
        print(f"Main loss: {val_metrics['main_loss']:.4f}")
        print(
            f"Auxiliary loss: "
            f"{val_metrics['auxiliary_loss']:.4f}"
        )

        if val_metrics["total_loss"] < best_val_loss:

            best_val_loss = val_metrics["total_loss"]

            checkpoint_path = output_dir / "best_model.pt"

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "max_length": args.max_length,
                    "validation_loss": best_val_loss,
                },
                checkpoint_path,
            )

            tokenizer.save_pretrained(
                output_dir / "tokenizer"
            )

            print("\nBest model saved:", checkpoint_path)

    print("\nTraining completed.")
    print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
    main()
