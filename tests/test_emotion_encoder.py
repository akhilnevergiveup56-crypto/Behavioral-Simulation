
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.models import emotion_encoder


class FakeEncoder(nn.Module):
    """Small transformer substitute for testing without downloads."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)

    def forward(self, input_ids, attention_mask):
        batch_size, sequence_length = input_ids.shape

        hidden_states = torch.randn(
            batch_size,
            sequence_length,
            self.config.hidden_size,
        )

        return SimpleNamespace(
            last_hidden_state=hidden_states
        )


def test_model_output_shapes(monkeypatch):
    # Prevent downloading the real pretrained model during testing.
    monkeypatch.setattr(
        emotion_encoder.AutoModel,
        "from_pretrained",
        lambda model_name: FakeEncoder(),
    )

    model = emotion_encoder.MultiTaskEmotionEncoder()

    input_ids = torch.ones((2, 128), dtype=torch.long)
    attention_mask = torch.ones((2, 128), dtype=torch.long)

    outputs = model(input_ids, attention_mask)

    assert outputs["representation"].shape == (2, 16)
    assert outputs["goemotions_logits"].shape == (2, 28)
    assert outputs["project_emotion_logits"].shape == (2, 8)
    assert outputs["project_emotions"].shape == (2, 8)


def test_project_emotion_predictions_are_probabilities(monkeypatch):
    monkeypatch.setattr(
        emotion_encoder.AutoModel,
        "from_pretrained",
        lambda model_name: FakeEncoder(),
    )

    model = emotion_encoder.MultiTaskEmotionEncoder()

    input_ids = torch.ones((2, 128), dtype=torch.long)
    attention_mask = torch.ones((2, 128), dtype=torch.long)

    outputs = model(input_ids, attention_mask)

    predictions = outputs["project_emotions"]

    assert torch.all(predictions >= 0)
    assert torch.all(predictions <= 1)


def test_multitask_loss_returns_three_losses():
    outputs = {
        "goemotions_logits": torch.randn(2, 28),
        "project_emotions": torch.sigmoid(torch.randn(2, 8)),
    }

    goemotions_targets = torch.zeros(2, 28)
    project_targets = torch.zeros(2, 8)
    project_mask = torch.ones(2, 8)

    losses = emotion_encoder.compute_multitask_loss(
        outputs,
        goemotions_targets,
        project_targets,
        project_mask,
    )

    assert set(losses.keys()) == {
        "total_loss",
        "main_loss",
        "auxiliary_loss",
    }

    for loss in losses.values():
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_masked_dimensions_do_not_receive_auxiliary_gradient():
    project_predictions = torch.full(
        (1, 8),
        0.8,
        requires_grad=True,
    )

    outputs = {
        "goemotions_logits": torch.zeros(1, 28),
        "project_emotions": project_predictions,
    }

    goemotions_targets = torch.zeros(1, 28)
    project_targets = torch.zeros(1, 8)

    # Only dimensions 0 and 2 have supervision.
    project_mask = torch.tensor([
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

    losses = emotion_encoder.compute_multitask_loss(
        outputs,
        goemotions_targets,
        project_targets,
        project_mask,
    )

    losses["auxiliary_loss"].backward()

    gradients = project_predictions.grad

    assert gradients[0, 0] != 0
    assert gradients[0, 2] != 0

    for index in [1, 3, 4, 5, 6, 7]:
        assert gradients[0, index] == 0


def test_all_zero_mask_produces_zero_auxiliary_loss():
    outputs = {
        "goemotions_logits": torch.randn(2, 28),
        "project_emotions": torch.sigmoid(torch.randn(2, 8)),
    }

    goemotions_targets = torch.zeros(2, 28)
    project_targets = torch.zeros(2, 8)
    project_mask = torch.zeros(2, 8)

    losses = emotion_encoder.compute_multitask_loss(
        outputs,
        goemotions_targets,
        project_targets,
        project_mask,
    )

    assert losses["auxiliary_loss"].item() == 0.0
    assert torch.isfinite(losses["total_loss"])
