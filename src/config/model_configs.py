from dataclasses import dataclass, field

@dataclass
class MLPConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout: float = 0.1

    hidden_dims: int = 1536
    residual: bool = False

@dataclass
class DinoFineTuneConfig:
    learning_rates: float = 1e-6
    weight_decays: float = 1e-5
    
@dataclass
class AdapterConfig:
    learning_rate: float = 1e-6
    weight_decay: float = 1e-5
    dropout: float = 0.1

    hidden_dims: int = 1536
    residual: bool = True

MLP_CONFIGS = [
    MLPConfig(learning_rate=lr, dropout=dropout)
    for lr in [1e-5, 1e-4, 1e-3]
    for dropout in [0.0, 0.1, 0.2]
]

ADAPTER_CONFIGS = [
    AdapterConfig(learning_rate=lr, dropout=dropout)
    for lr in [1e-6, 1e-5, 1e-4]
    for dropout in [0.0, 0.1, 0.2]
]