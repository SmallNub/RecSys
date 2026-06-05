import numpy as np

from src.models import SpecialTokens

from .augment import CropAndAugment


class SASRecPrePro:
    def __init__(
        self,
        *,
        items_ref,
        split,  # One of ["train", "valid", "test"]
        # Timeline crop / augment
        crop_length,
        augment,
        random_crop=True,
        **kwargs,
    ):
        self.split = split
        # Context parameters
        self.crop_and_augment = CropAndAugment(
            crop_length=crop_length,
            augment=augment,
            random_crop=random_crop,
            split=split,
        )

        items_map = items_ref
        self.item_to_id = items_map["item_to_id"]
        self.id_to_item = items_map["id_to_item"]

    def __call__(self, row):
        # Quantize the timeline
        tl, _ = self.crop_and_augment(row["timeline"], row["timestamp"])

        # Convert to ids
        tl = [self.item_to_id[item] for item in tl]

        pad_to = self.crop_and_augment.crop_length

        query = [SpecialTokens.PAD.value] * (pad_to - len(tl[:-1])) + tl[:-1]
        target = [SpecialTokens.PAD.value] * (pad_to - len(tl[1:])) + tl[1:]

        return {
            "query": np.array(query),
            "target": np.array(target),
        }
