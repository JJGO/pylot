import torch

from ..util import printc


def print_hook(module, inputs, outputs):
    printc(module, color="RED")
    printc(inputs[0].shape, color='GRASS')
    printc(outputs.shape, color='YELLOW')
    print()


def forward_verbose(model, input):

    hooks = []

    def register_print(module):
        hooks.append(module.register_forward_hook(print_hook))

    model.apply(register_print)

    with torch.no_grad():
        model(input)

    for h in hooks:
        h.remove()
