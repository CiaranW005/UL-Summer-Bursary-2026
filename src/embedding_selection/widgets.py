from dataclasses import dataclass
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display

from typing import cast, Literal

UNSELECTED: Literal["UNSELECTED"] = "UNSELECTED"
ModelPath = Path | Literal["UNSELECTED"]

@dataclass
class ModelSelector:
    model_selector: widgets.Dropdown
    confirm_button: widgets.Button
    output: widgets.Output

    filter: widgets.RadioButtons

    anomaly_block: list[Path]
    anomaly_head: list[Path]

    category_head: list[Path]
    fine_tuned_dino: list[Path]

    confirmed: bool = False
    selected_path: ModelPath = UNSELECTED

    def __post_init__(self) -> None:
        self.filter.observe(self._update_options, names="value")

        self.model_selector.observe(self._invalidate_selection, names="value")

        self.confirm_button.on_click(self._confirm_selection)

        self._update_options()

    def _update_options(self, change: dict[str, object] | None = None) -> None:
        filter = cast(str, self.filter.value)

        if filter == "All Models":
            paths = (
                self.anomaly_block +
                self.anomaly_head +
                self.category_head +
                self.fine_tuned_dino
            )
        
        elif filter == "Anomaly Detection Models":
            paths = (
                self.anomaly_block +
                self.anomaly_head
            )

        elif filter == "Category Head Models":
            paths = self.category_head

        elif filter == "DINO Models":
            paths = (self.anomaly_block + self.fine_tuned_dino)

        else:
            raise ValueError(f"Unknown model filter: {filter}")

        self.model_selector.options = [
            ("Select a model", UNSELECTED),
            *[
                (path.name, path)
                for path in paths
            ]
        ]

        self.model_selector.value = UNSELECTED
        self._invalidate_selection()

    def _confirm_selection(self, _: widgets.Button) -> None:
        self.output.clear_output()

        with self.output:
            selected = cast(ModelPath, self.model_selector.value)

            if selected == UNSELECTED:
                print("Select a model first")
                return

            self.selected_path = selected

        self.confirmed = True
        print("Selection Confirmed")

    def _invalidate_selection(self, change: dict[str, object] | None = None) -> None:
        self.confirmed = False
        self.selected_path = UNSELECTED

    def get_selected_path(self) -> Path:
        if not self.confirmed:
            raise RuntimeError(
                "Select and confirm a model first."
            )

        if self.selected_path == UNSELECTED:
            raise RuntimeError("No model selected.")

        return self.selected_path

    def display(self) -> None:
        display(
            self.filter,
            self.model_selector,
            self.confirm_button,
            self.output,
        )


def create_model_selector(
        *,
        adapter_block_checkpoints: list[Path],
        anomaly_head_checkpoints: list[Path],
        category_head_checkpoints: list[Path],
        fine_tuned_dino_checkpoints: list[Path]
)-> ModelSelector:
    filter = widgets.RadioButtons(
        options=[
            "All Models",
            "Anomaly Detection Models",
            "Category Head Models",
            "DINO Models"
        ],
        description="Model Types:"
    )
    model_selector = widgets.Dropdown(
        options=[
        ("Select a model", UNSELECTED),
        ],
        value=UNSELECTED,
        description="Model:",
        layout=widgets.Layout(width="800px")
    )

    confirm_button = widgets.Button(
        description="confirm_selection",
        button_style="success",
        layout=widgets.Layout(width="180px")
    )

    ouput = widgets.Output()

    return ModelSelector(
        model_selector=model_selector,
        confirm_button=confirm_button,
        output=ouput,
        filter=filter,
        anomaly_block=adapter_block_checkpoints,
        anomaly_head=anomaly_head_checkpoints,
        category_head=category_head_checkpoints,
        fine_tuned_dino=fine_tuned_dino_checkpoints
    )