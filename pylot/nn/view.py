from torch import nn


class View(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

    def __repr__(self):
        return f'View({self.shape})'
