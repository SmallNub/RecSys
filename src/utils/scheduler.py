import math

from torch.optim.lr_scheduler import LRScheduler


class CosineScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        cosine_steps,
        warmup_factor=1e-1,
        final_factor=1e-1,
    ):
        self.optimizer = optimizer

        self.warmup_factor = warmup_factor
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps
        self.final_factor = final_factor

        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_sizes = [
            (base_lr - self.warmup_factor * base_lr) / warmup_steps
            for base_lr in self.base_lrs
        ]

        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch == 0:
            # Start at warm * lr
            return [base_lr * self.warmup_factor for base_lr in self.base_lrs]

        elif self.last_epoch < self.warmup_steps:
            # Linear warmup from warm*lr to lr
            return [
                base_lr * self.warmup_factor + step_size * self.last_epoch
                for base_lr, step_size in zip(self.base_lrs, self.step_sizes)
            ]

        elif self.last_epoch < self.warmup_steps + self.cosine_steps:
            return [
                base_lr * self.final_factor
                # Apply cosine decay formula from warmup peak down to the final minimum learning rate factor
                + 0.5
                * (base_lr - base_lr * self.final_factor)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.last_epoch - self.warmup_steps)
                        / self.cosine_steps
                    )
                )
                for base_lr in self.base_lrs
            ]

        else:
            return [base_lr * self.final_factor for base_lr in self.base_lrs]
