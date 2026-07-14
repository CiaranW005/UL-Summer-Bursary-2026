from dataclasses import dataclass
import pandas as pd 

@dataclass
class CategoryMasks:
    category: str
    
    train_mask: pd.Series
    test_mask: pd.Series
    train_category_mask: pd.Series

    good_test_mask: pd.Series
    defect_test_mask: pd.Series

def get_category_masks(meta, category):
    train_mask = meta["split"] == "train"
    test_mask = ~train_mask

    train_meta = meta[train_mask]
    test_meta = meta[test_mask]

    train_cat_mask = train_meta["category"] == category

    good_test_mask = (
        (test_meta["category"] == category)
        & (test_meta["type"] == "good")
    )

    defect_test_mask = (
        (test_meta["category"] == category)
        & (test_meta["type"] != "good")
    )

    return CategoryMasks(
        category=category,
        train_mask=train_mask,
        test_mask=test_mask,
        train_category_mask=train_cat_mask,
        good_test_mask=good_test_mask,
        defect_test_mask=defect_test_mask
    )