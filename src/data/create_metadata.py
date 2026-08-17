import os
import pandas as pd
import numpy as np

from PIL import Image
from pathlib import Path

from ..config.paths import ROOT, DATA_DIR, DATASET_DIR, CSV_PATH
from .types import ImageRecord

def create_metadata() -> None:
    """
    
    """
    images: list[ImageRecord] = []

    # Go through each category in the dataset
    for cat in DATASET_DIR.iterdir():
        if not cat.is_dir():
            continue

        #print(cat.name)

        # Within each category, go through each split (train/test)
        for split in cat.iterdir():
            if not split.is_dir():
                continue
            
            # Not interested in having the masks within the metadata
            if split.name == "ground_truth":
                continue
            
            #print(f"    {split.name}")

            # Within each split, go through each type (good/defect)
            for defect_type in split.iterdir():    
                #print(f"        {type.name}")

                for img in defect_type.iterdir():
                    #print(f"            {img.name}")

                    if defect_type.name == "good":
                        defect_coverage = 0.0
                    else:
                        mask_path = (cat / "ground_truth" / defect_type.name / f"{img.stem}_mask.png")

                        mask = np.asarray(Image.open(mask_path))
                        defect_coverage = np.count_nonzero(mask) / mask.size

                    image_record : ImageRecord = {
                        "path": str(img.relative_to(ROOT)),
                        "category": cat.name,
                        "split": split.name,
                        "type": defect_type.name,
                        "label": 0 if defect_type.name == "good" else 1,
                        "defect_coverage": defect_coverage
                    }

                    images.append(image_record)

    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.DataFrame(images)

    df.to_csv(CSV_PATH, index=False)
    