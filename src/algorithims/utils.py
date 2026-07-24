import pandas as pd 

from .types import CategoryMasks

def get_category_masks(meta: pd.DataFrame, category: str) -> CategoryMasks:
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
