
import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from src.data.loader import create_goemotions_dataloader
from src.models.emotion_encoder import (
    MultiTaskEmotionEncoder,
    compute_multitask_loss,
)
from src.training.class_weights import (
    compute_goemotions_pos_weights,
)


def set_seed(seed=42):
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Select the best available training device."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def limit_dataloader(
    loader,
    max_samples=None,
    shuffle=False,
    seed=42,
):
    """Select a reproducible random subset of the dataset."""

    if max_samples is None:
        return loader

    sample_count = min(
        max_samples,
        len(loader.dataset),
    )

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(loader.dataset),
        generator=generator,
    )[:sample_count].tolist()

    subset = Subset(
        loader.dataset,
        indices,
    )

    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def run_epoch(
    model,
    dataloader,
    device,
    optimizer=None,
    goemotions_pos_weight=None,
):
    """Run one training or validation epoch."""

    training = optimizer is not None
    model.train(training)

    total_loss_sum = 0.0
    main_loss_sum = 0.0
    auxiliary_loss_sum = 0.0

    example_count = 0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

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

            # Apply class weights during training only.
            # Keep validation loss unweighted.
            current_pos_weight = (
                goemotions_pos_weight
                if training
                else None
            )

            losses = compute_multitask_loss(
                outputs=outputs,
                goemotions_targets=goemotions_targets,
                project_emotion_targets=project_targets,
                project_emotion_mask=project_mask,
                auxiliary_weight=0.3,
                goemotions_pos_weight=current_pos_weight,
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

        total_loss_sum += (
            loss.item() * batch_size
        )

        main_loss_sum += (
            losses["main_loss"].item() * batch_size
        )

        auxiliary_loss_sum += (
            losses["auxiliary_loss"].item() * batch_size
        )

        example_count += batch_size

    if example_count == 0:
        raise ValueError(
            "DataLoader produced no examples."
        )

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

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive.")

    if args.learning_rate <= 0 or args.max_length < 1:
        parser.error("Learning rate and max length must be positive.")

    if (
        args.max_train_samples is not None
        and args.max_train_samples < 1
    ):
        parser.error("Maximum training samples must be positive.")

    if (
        args.max_val_samples is not None
        and args.max_val_samples < 1
    ):
        parser.error("Maximum validation samples must be positive.")

    set_seed(args.seed)

    # Project directories
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

    # Keep this experiment separate from the previous model.
    output_dir = (
        project_root
        / "checkpoints"
        / "goemotions-distilbert-weighted"
    )

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    if not val_path.exists():
        raise FileNotFoundError(val_path)

    device = get_device()

    print("Training device:", device)
    print("Random seed:", args.seed)

    # Calculate class weights using training data only.
    print("\nCalculating emotion class weights...")

    goemotions_pos_weight = compute_goemotions_pos_weights(
        train_path,
        max_weight=10.0,
    ).to(device)

    print(
        "Number of class weights:",
        len(goemotions_pos_weight),
    )

    print(
        "Minimum class weight:",
        round(goemotions_pos_weight.min().item(), 4),
    )

    print(
        "Maximum class weight:",
        round(goemotions_pos_weight.max().item(), 4),
    )

    # Load tokenizer
    model_name = "distilbert-base-uncased"

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create DataLoaders
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

    # Optionally select reproducible random subsets.
    train_loader = limit_dataloader(
        train_loader,
        max_samples=args.max_train_samples,
        shuffle=True,
        seed=args.seed,
    )

    val_loader = limit_dataloader(
        val_loader,
        max_samples=args.max_val_samples,
        shuffle=False,
        seed=args.seed,
    )

    print(
        "Training examples:",
        len(train_loader.dataset),
    )

    print(
        "Validation examples:",
        len(val_loader.dataset),
    )

    # Load model
    print("\nLoading emotion model...")

    model = MultiTaskEmotionEncoder(
        model_name=model_name,
        num_goemotions=28,
        num_project_emotions=8,
    ).to(device)

    # Optimizer
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

    # Training loop
    for epoch in range(args.epochs):

        print(
            f"\n========== EPOCH {epoch + 1}/{args.epochs} =========="
        )

        # Training uses weighted loss.
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            goemotions_pos_weight=goemotions_pos_weight,
        )

        # Validation uses unweighted loss.
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            optimizer=None,
            goemotions_pos_weight=None,
        )

        # Display training results.
        print("\nTraining metrics (weighted):")

        print(
            f"Total loss: {train_metrics['total_loss']:.4f}"
        )

        print(
            f"Main loss: {train_metrics['main_loss']:.4f}"
        )

        print(
            f"Auxiliary loss: {train_metrics['auxiliary_loss']:.4f}"
        )

        # Display validation results.
        print("\nValidation metrics (unweighted):")

        print(
            f"Total loss: {val_metrics['total_loss']:.4f}"
        )

        print(
            f"Main loss: {val_metrics['main_loss']:.4f}"
        )

        print(
            f"Auxiliary loss: {val_metrics['auxiliary_loss']:.4f}"
        )

        # Preserve the checkpoint with the best validation
        # main loss.
        if val_metrics["main_loss"] < best_val_loss:

            best_val_loss = val_metrics["main_loss"]

            checkpoint_path = (
                output_dir / "best_model.pt"
            )

            cpu_state_dict = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": cpu_state_dict,
                    "model_name": model_name,
                    "max_length": args.max_length,
                    "validation_main_loss": best_val_loss,
                    "validation_total_loss": val_metrics["total_loss"],
                    "class_weights": goemotions_pos_weight.cpu(),
                    "seed": args.seed,
                },
                checkpoint_path,
            )

            tokenizer.save_pretrained(
                output_dir / "tokenizer"
            )

            print(
                "\nBest model saved:",
                checkpoint_path,
            )

    print("\nTraining completed.")

    print(
        "Best validation main loss:",
        best_val_loss,
    )

    print(
        "Checkpoint directory:",
        output_dir,
    )


if __name__ == "__main__":
    main()
