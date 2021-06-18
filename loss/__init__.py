from .vae import vae_loss, kld_loss
from .total_variation import total_variation2d
from .flat import *
from .multi import MultiLoss
from .segmentation import (
    soft_jaccard_score,
    soft_dice_loss,
    SoftDiceLoss,
    SoftJaccardLoss,
    PixelMSELoss
)
