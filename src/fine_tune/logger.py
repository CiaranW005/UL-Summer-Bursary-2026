import wandb

import os
from dotenv import load_dotenv

from .types import ModelInfo

load_dotenv()

class TrainLogger:
    def __init__(
            self,
            project: str,
            name: str,
            model_info: ModelInfo
        ) -> None:

        self.run = wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=project,
            name=name,
            config=dict(model_info)
        )

    def log(self, data: dict[str, int | float]) -> None:
        self.run.log(data)

    def finish(self) -> None:
        self.run.finish()