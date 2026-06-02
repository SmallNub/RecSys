import numpy as np
import ray

from src.models import SpecialTokens

from .augment import CropAndAugment


class Remap:
    def __init__(self, L, K):
        self.L = L
        self.K = K

        self.levels_offset = np.arange(L) * K  # Broadcasted
        self.special_tokens_offset = len(SpecialTokens)

    def __call__(self, quantized):
        return quantized + self.levels_offset + self.special_tokens_offset


class MARIUSPrePro:
    def __init__(
        self,
        *,  # Pass all arguments as keyword arguments
        quantizer_ref,
        split,
        crop_length,
        augment,
        random_crop=True,
        **kwargs,
    ):
        self.split = split

        # Context parameters
        self.split = split
        # Context parameters
        self.crop_and_augment = CropAndAugment(
            crop_length=crop_length,
            augment=augment,
            random_crop=random_crop,
            split=split,
        )

        self.quant_df = ray.get(quantizer_ref)

        self.L = len(self.quant_df.columns)
        self.K = self.quant_df.values.max() + 1
        print(f"Quantized table has {self.L} levels and {self.K} codes per level.")

        self.remap = Remap(L=self.L, K=self.K)

    def __call__(self, row):
        # Quantize the timeline
        tl, ts = self.crop_and_augment(row["timeline"], row["timestamp"])

        quantized_query = self.remap(self.quant_df.loc[tl].values)
        # Shape : L x K

        pad_to = self.crop_and_augment.crop_length

        # Target - All items except first.
        target = quantized_query[1:]
        target = np.concatenate(
            [
                np.full(
                    (pad_to - target.shape[0], self.L),
                    -100,
                ),
                target,
            ],
            axis=0,
        )

        # Prepare the input - Drop last item
        input = quantized_query[:-1]
        input = np.concatenate(
            [
                np.full(
                    (pad_to - input.shape[0] - 1, self.L),
                    SpecialTokens.PAD.value,
                ),
                np.full((1, self.L), SpecialTokens.BOS.value),
                input,
            ],
            axis=0,
        )

        return {"input": input, "target": target}
