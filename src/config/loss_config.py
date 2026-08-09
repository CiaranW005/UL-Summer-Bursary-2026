from dataclasses import dataclass, field

@dataclass
class UnsupervisedLossConfig:
    category_weight: float
    vicreg_weight: float
    preservation_weight: float

UNSUPERVISED_LOSS_CONFIGS = [
    UnsupervisedLossConfig(
        category_weight=1.0,
        vicreg_weight=0.0,
        preservation_weight=0.0,
    ),
    UnsupervisedLossConfig(
        category_weight=0.0,
        vicreg_weight=1.0,
        preservation_weight=0.0,
    ),
    UnsupervisedLossConfig(
        category_weight=1.0,
        vicreg_weight=1.0,
        preservation_weight=0.0,
    ),
    UnsupervisedLossConfig(
        category_weight=1.0,
        vicreg_weight=0.0,
        preservation_weight=1.0,
    ),
    UnsupervisedLossConfig(
        category_weight=0.0,
        vicreg_weight=1.0,
        preservation_weight=1.0,
    ),
    UnsupervisedLossConfig(
        category_weight=1.0,
        vicreg_weight=1.0,
        preservation_weight=1.0,
    ),
]

@dataclass
class AnomalyLossConfig:
    anomaly_weight: float
    vicreg_weight: float
    
ANOMALY_LOSS_CONFIGS = [
    AnomalyLossConfig(1.0, 0.0),
    AnomalyLossConfig(1.0, 0.25),
    AnomalyLossConfig(1.0, 0.5),
    AnomalyLossConfig(1.0, 1.0),
    AnomalyLossConfig(0.5, 1.0),
    AnomalyLossConfig(0.25, 1.0),
]