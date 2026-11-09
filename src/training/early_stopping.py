
import math


class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    patience:
        Number of consecutive epochs without improvement
        before training stops.

    min_delta:
        Minimum improvement required to count as progress.
    """

    def __init__(self, patience=3, min_delta=0.001):

        if patience < 1:
            raise ValueError("Patience must be at least 1.")

        if min_delta < 0:
            raise ValueError("min_delta cannot be negative.")

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.bad_epochs = 0

    def step(self, validation_loss):
        """
        Update early stopping using the latest validation loss.

        Returns:
            improved: Whether validation loss improved.
            should_stop: Whether training should stop.
        """

        if not math.isfinite(validation_loss):
            raise ValueError("Validation loss must be finite.")

        improved = (
            validation_loss < self.best_loss - self.min_delta
        )

        if improved:
            self.best_loss = validation_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        should_stop = self.bad_epochs >= self.patience

        return improved, should_stop
