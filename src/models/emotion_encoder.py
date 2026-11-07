
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


class MultiTaskEmotionEncoder(nn.Module):
    """
    Shared transformer encoder with two prediction heads.

    Head 1: Predicts 28 GoEmotions categories.
    Head 2: Predicts 8 project emotion alignment scores.
    """

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        num_goemotions=28,
        num_project_emotions=8,
        dropout=0.2,
    ):
        super().__init__()

        # Load the pretrained transformer
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Main prediction head: 28 GoEmotions labels
        self.goemotions_head = nn.Linear(
            hidden_size,
            num_goemotions,
        )

        # Auxiliary prediction head: 8 project dimensions
        self.project_emotions_head = nn.Linear(
            hidden_size,
            num_project_emotions,
        )

    def forward(self, input_ids, attention_mask):
        # Convert text tokens into contextual representations
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use the first token representation
        representation = encoder_outputs.last_hidden_state[:, 0, :]

        representation = self.dropout(representation)

        # Predict 28 GoEmotions categories
        goemotions_logits = self.goemotions_head(
            representation
        )

        # Predict 8 project emotion alignment scores
        project_emotion_logits = self.project_emotions_head(
            representation
        )

        project_emotions = torch.sigmoid(
            project_emotion_logits
        )

        return {
            "representation": representation,
            "goemotions_logits": goemotions_logits,
            "project_emotion_logits": project_emotion_logits,
            "project_emotions": project_emotions,
        }


def compute_multitask_loss(
    outputs,
    goemotions_targets,
    project_emotion_targets,
    project_emotion_mask,
    auxiliary_weight=0.3,
):
    """
    Calculate the combined main and masked auxiliary losses.

    The mask prevents unsupported project dimensions
    from contributing to the auxiliary loss.
    """

    # Main loss: multi-label classification over 28 categories
    main_loss = F.binary_cross_entropy_with_logits(
        outputs["goemotions_logits"],
        goemotions_targets,
    )

    # Calculate auxiliary loss separately for each dimension
    auxiliary_elementwise_loss = F.mse_loss(
        outputs["project_emotions"],
        project_emotion_targets,
        reduction="none",
    )

    # Ignore dimensions that do not have valid supervision
    masked_auxiliary_loss = (
        auxiliary_elementwise_loss * project_emotion_mask
    )

    # Normalize by the number of supervised targets
    valid_targets = project_emotion_mask.sum().clamp_min(1.0)

    auxiliary_loss = (
        masked_auxiliary_loss.sum() / valid_targets
    )

    # Combine both objectives
    total_loss = (
        main_loss + auxiliary_weight * auxiliary_loss
    )

    return {
        "total_loss": total_loss,
        "main_loss": main_loss,
        "auxiliary_loss": auxiliary_loss,
    }
