from torch.nn import (
        AdaptiveLogSoftmaxWithLoss,
        BCELoss,
        BCEWithLogitsLoss,
        CTCLoss,
        CosineEmbeddingLoss,
        CrossEntropyLoss,
        HingeEmbeddingLoss,
        KLDivLoss,
        L1Loss,
        MSELoss,
        MarginRankingLoss,
        MultiLabelMarginLoss,
        MultiLabelSoftMarginLoss,
        MultiMarginLoss,
        NLLLoss,
        PoissonNLLLoss,
        SmoothL1Loss,
        SoftMarginLoss,
        TripletMarginLoss,
)
from .vae import vae_loss, kld_loss
from .total_variation import total_variation2d
from .flat import flatten_loss
