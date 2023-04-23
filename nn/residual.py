from pydantic import validate_arguments

from torch import nn


class ConvResidual(nn.Module):
    @validate_arguments
    def __init__(
        self, module, in_channels: int, out_channels: int,
    ):
        super().__init__()
        self.main = module
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            # TODO do we want to init these to 1, like controlnet's zeroconv
            # TODO do we want to initialize these like the other conv layers
            # TODO Drop path (block)
            # TODO Drop path (sample)
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )

    def forward(self, input):
        return self.main(input) + self.shortcut(input)
