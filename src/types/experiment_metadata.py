from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    K_frac : float
    start_growth : float
    min_growth : float
    reg : float | None = None
    growth_type : str | None = None
    cleaner : str | None = None


@dataclass
class AlgorithmResults:
    config: ExperimentConfig

    category: str
    n_shapes: int
    auroc : float

    normal_inside: int
    defect_inside : int
