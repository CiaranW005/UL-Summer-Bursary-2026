from dataclasses import dataclass
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display

from typing import cast, Literal

UNSELECTED: Literal["UNSELECTED"] = "UNSELECTED"

@dataclass
class ModelSelector:
    dino_selector: widgets.Dropdown
    confirm_button: widgets.Button
    output: widgets.Output

    use_category_head: widgets.Checkbox | None = None
    category_head_selector: widgets.Dropdown | None = None

    confirmed: bool = False
    selected_path: Path | None | Literal["UNSELECTED"] = UNSELECTED

    def __post_init__(self) -> None:
        self.confirm_button.on_click(self._confirm_selection) # pyright: ignore[reportUnknownMemberType]

        self.dino_selector.observe(
            self._invalidate_selection,
            names="value"
        )

        if self.use_category_head is not None:
            self.use_category_head.observe(
                self._update_visibility,
                names="value"
            )

        if self.category_head_selector is not None:
            self.category_head_selector.observe(
                self._invalidate_selection,
                names="value"
            )

        self._update_visibility()

    def _update_visibility(self, change: dict[str, object] | None = None) -> None:
        if self.use_category_head is None or self.category_head_selector is None:
            return
        
        if self.use_category_head.value:
            self.dino_selector.layout.display = "none" # pyright: ignore[reportUnknownMemberType]
            self.category_head_selector.layout.display = "" # pyright: ignore[reportUnknownMemberType]
        else:
            self.dino_selector.layout.display = "" # pyright: ignore[reportUnknownMemberType]
            self.category_head_selector.layout.display = "none" # pyright: ignore[reportUnknownMemberType]

    def _confirm_selection(self, _: widgets.Button) -> None:
        self.output.clear_output()  # pyright: ignore[reportUnknownMemberType]

        with self.output:
            use_category = (self.use_category_head is not None and self.use_category_head.value)

            if use_category:
                if self.category_head_selector is None:
                    raise RuntimeError("Category head selector is missing.")
                
                selected = cast(Path | None | Literal["UNSELECTED"], self.category_head_selector.value) # pyright: ignore[reportUnknownMemberType]

                if selected is None:
                    print("Select a category head first.")
                    return

                self.selected_path = selected
            else:
                selected = cast(Path | None | Literal["UNSELECTED"], self.dino_selector.value) # pyright: ignore[reportUnknownMemberType]

                if selected == UNSELECTED:
                    print("Select a DINO model first.")
                    return

                self.selected_path = selected

            self.confirmed = True
            print("Selection confirmed.")

    def _invalidate_selection(self, change: dict[str, object] | None = None) -> None:
        self.confirmed = False
        self.selected_path = UNSELECTED

    def display(self) -> None:
        items: list[widgets.Widget] = []

        if self.use_category_head is not None:
            items.append(self.use_category_head)

        items.append(self.dino_selector)

        if self.category_head_selector is not None:
            items.append(self.category_head_selector)

        items.extend([self.confirm_button, self.output])
        
        display(*items)

    def get_selected_path(self) -> Path | None:
        if not self.confirmed:
            raise RuntimeError(
                "Choose a model and confirm the selection' first."
            )

        if self.selected_path == UNSELECTED:
            raise RuntimeError("No model selected.")

        return self.selected_path

def create_model_selector(
    *,
    dino_checkpoints: list[Path],
    category_head_checkpoints: list[Path] | None = None,
) -> ModelSelector:
    dino_selector = widgets.Dropdown(
        options=[
            ("Select a DINO model", UNSELECTED),
            ("Pretrained DINOv2 ViT-S/14", None),
            *[
                (path.name, path)
                for path in dino_checkpoints
            ],
        ],
        value=UNSELECTED,
        description="DINO:",
        layout=widgets.Layout(width="800px"),
    )

    confirm_button = widgets.Button(
        description="Confirm selection",
        button_style="success",
        layout=widgets.Layout(width="180px"),
    )

    output = widgets.Output()

    if category_head_checkpoints is None:
        return ModelSelector(
            dino_selector=dino_selector,
            confirm_button=confirm_button,
            output=output,
        )

    return ModelSelector(
        dino_selector=dino_selector,
        confirm_button=confirm_button,
        output=output,
        category_head_selector=widgets.Dropdown(
            options=[
                ("Select a category head", None),
                *[
                    (path.name, path)
                    for path in category_head_checkpoints
                ],
            ],
            value=None,
            description="Category head:",
            layout=widgets.Layout(width="800px"),
        ),
        use_category_head=widgets.Checkbox(
            value=False,
            description="Use category head",
        )
    )