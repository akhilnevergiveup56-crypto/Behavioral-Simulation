
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


class MultiTaskEmotionEncoder(nn.Module):
    """
    Shared transformer encoder with two prediction heads.

    Head 1:
        Predicts 28 GoEmotions categories.

    Head 2:
        Predicts 8 project emotion alignment scores.
    """

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        num_goemotions=28,
        num_project_emotions=8,
        dropout=0.2,
    ):
        super().__init__()

        # Load the pretrained transformer encoder.
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Main prediction head: 28 GoEmotions labels.
        self.goemotions_head = nn.Linear(
            hidden_size,
            num_goemotions,
        )

        # Auxiliary prediction head: 8 project dimensions.
        self.project_emotions_head = nn.Linear(
            hidden_size,
            num_project_emotions,
        )

    def forward(self, input_ids, attention_mask):
        """
        Convert input text into emotion predictions.
        """

        # Generate contextual text representations.
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use the first token representation.
        representation = (
            encoder_outputs.last_hidden_state[:, 0, :]
        )

        representation = self.dropout(representation)

        # Predict the original 28 emotion categories.
        goemotions_logits = self.goemotions_head(
            representation
        )

        # Predict the 8 project emotion dimensions.
        project_emotion_logits = self.project_emotions_head(
            representation
        )

        # Convert auxiliary logits into values between 0 and 1.
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
    goemotions_pos_weight=None,
):
    """
    Calculate the combined main and auxiliary losses.

    Parameters
    ----------
    outputs:
        Predictions returned by MultiTaskEmotionEncoder.

    goemotions_targets:
        Ground-truth multi-hot vectors for 28 emotions.

    project_emotion_targets:
        Eight project emotion alignment targets.

    project_emotion_mask:
        Indicates which project dimensions have valid
        mapping supervision.

    auxiliary_weight:
        Controls the contribution of the auxiliary loss.

    goemotions_pos_weight:
        Optional tensor containing 28 positive-class weights.
        Higher values give more importance to positive examples
        from underrepresented emotion categories.
    """

    goemotions_logits = outputs["goemotions_logits"]

    # Validate auxiliary loss weight.
    if auxiliary_weight < 0:
        raise ValueError(
            "Auxiliary weight cannot be negative."
        )

    # Validate optional class weights.
    if goemotions_pos_weight is not None:

        if goemotions_pos_weight.shape != (28,):
            raise ValueError(
                "Expected exactly 28 positive-class weights."
            )

        if not torch.isfinite(goemotions_pos_weight).all():
            raise ValueError(
                "Class weights must contain finite values."
            )

        if (goemotions_pos_weight <= 0).any():
            raise ValueError(
                "Class weights must be positive."
            )

        # Move weights to the same device and dtype as the logits.
        goemotions_pos_weight = goemotions_pos_weight.to(
            device=goemotions_logits.device,
            dtype=goemotions_logits.dtype,
        )

    # -------------------------------------------------
    # 1. Main loss: 28-label GoEmotions classification
    # -------------------------------------------------

    main_loss = F.binary_cross_entropy_with_logits(
        goemotions_logits,
        goemotions_targets,
        pos_weight=goemotions_pos_weight,
    )

    # -------------------------------------------------
    # 2. Auxiliary loss: 8 project emotion dimensions
    # -------------------------------------------------

    auxiliary_elementwise_loss = F.mse_loss(
        outputs["project_emotions"],
        project_emotion_targets,
        reduction="none",
    )

    # Ignore dimensions without valid supervision.
    masked_auxiliary_loss = (
        auxiliary_elementwise_loss
        * project_emotion_mask
    )

    # Normalize using only supervised dimensions.
    valid_targets = project_emotion_mask.sum().clamp_min(1.0)

    auxiliary_loss = (
        masked_auxiliary_loss.sum()
        / valid_targets
    )

    # -------------------------------------------------
    # 3. Combined multitask loss
    # -------------------------------------------------

    total_loss = (
        main_loss
        + auxiliary_weight * auxiliary_loss
    )

    return {
        "total_loss": total_loss,
        "main_loss": main_loss,
        "auxiliary_loss": auxiliary_loss,
    }
