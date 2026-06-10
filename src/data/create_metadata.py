import os
import pandas as pd

from ..config import ROOT, DATA_DIR, DATASET_DIR, CSV_PATH

def create_metadata():
    images = []

    # Go through each category in the dataset
    for cat in DATASET_DIR.iterdir():
        if not cat.is_dir():
            continue

        #print(cat.name)

        # Within each category, go through each split (train/test)
        for split in cat.iterdir():
            if not split.is_dir():
                continue
            
            # Ignore ground truth for now as it's not what we're interested in
            if split.name == "ground_truth":
                continue
            
            #print(f"    {split.name}")

            # Within each split, go through each type (good/defect)
            for type in split.iterdir():    
                #print(f"        {type.name}")

                for img in type.iterdir():
                    #print(f"            {img.name}")

                    image_record = {
                        "path": str(img.relative_to(ROOT)),
                        "category": cat.name,
                        "split": split.name,
                        "type": type.name,
                        "label": 0 if type.name == "good" else 1
                    }

                    images.append(image_record)

    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.DataFrame(images)

    df.to_csv(CSV_PATH, index=False)
    