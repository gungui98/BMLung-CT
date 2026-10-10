"""Loss functions."""
from bmct.losses.dice import soft_dice_loss
from bmct.losses.masked_multitask import MaskedMultitaskLoss

__all__ = ["soft_dice_loss", "MaskedMultitaskLoss"]
