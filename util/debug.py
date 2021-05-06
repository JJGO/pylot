import torch


def torch_traceback():

    from rich.traceback import install
    install(show_locals=True)
    def repr(self):
        return f"Tensor<{', '.join(map(str, self.shape))}>"
    torch.Tensor.__repr__ = repr
