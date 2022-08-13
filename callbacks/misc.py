import torch


class LockMemory:
    def __init__(self, experiment, gigs):
        self.vals = []
        for g in range(gigs):
            x = torch.randint(
                0,
                128,
                size=(1024, 1024, 1024),
                dtype=torch.uint8,
                device=experiment.device,
            )
            self.vals.append(x)

    def __call__(self, *args, **kwargs):
        pass
