from ...fine_tune.losses.types import LossCollection, LossName, WeightedLoss
from ...fine_tune.losses.unsupervised_head import CategoryContLoss, VICRegLoss, PreservationLoss
from ...fine_tune.losses.anomaly_head import CategoryAnomalyContrastiveLoss
from ...config.loss_config import UnsupervisedLossConfig, AnomalyLossConfig

def build_unsupervised_losses(config: UnsupervisedLossConfig) -> LossCollection:
    losses = []

    if config.category_weight > 0:
        losses.append(
            WeightedLoss(
                name=LossName.CATEGORY_CONTRASTIVE,
                loss=CategoryContLoss(
                    temperature=0.1,
                ),
                weight=config.category_weight,
            )
        )

    if config.vicreg_weight > 0:
        losses.append(
            WeightedLoss(
                name=LossName.VICREG,
                loss=VICRegLoss(
                    invar_weight=25,
                    var_weight=25,
                    cov_weight=1,
                    var_target=1,
                ),
                weight=config.vicreg_weight,
            )
        )

    if config.preservation_weight > 0:
        losses.append(
            WeightedLoss(
                name=LossName.PRESERVATION,
                loss=PreservationLoss(
                    sim_weight=1.0,
                    norm_weight=1.0,
                ),
                weight=config.preservation_weight,
            )
        )

    return LossCollection(losses)

def build_anomaly_losses(config: AnomalyLossConfig) -> LossCollection:
    losses = [
        WeightedLoss(
            name=LossName.ANOMALY_CATEGORY_CONTRASTIVE,
            loss=CategoryAnomalyContrastiveLoss(
                temperature=0.1,
            ),
            weight=config.anomaly_weight,
        )
    ]

    if config.vicreg_weight > 0:
        losses.append(
            WeightedLoss(
                name=LossName.VICREG,
                loss=VICRegLoss(
                    invar_weight=25,
                    var_weight=25,
                    cov_weight=1,
                    var_target=1,
                ),
                weight=config.vicreg_weight,
            )
        )

    return LossCollection(losses)