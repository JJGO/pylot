import numpy as np
import torch
from torch import nn

from .size import dtype2bits
from ..pruning.utils import get_activations


def hook_applyfn(hook, model, forward=False, backward=False):
    """

    [description]

    Arguments:
        hook {[type]} -- [description]
        model {[type]} -- [description]

    Keyword Arguments:
        forward {bool} -- [description] (default: {False})
        backward {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    assert forward ^ backward, \
        "Either forward or backward must be True"
    hooks = []

    def register_hook(module):
        if (
            not isinstance(module, nn.Sequential)
            and
            not isinstance(module, nn.ModuleList)
            and
            not isinstance(module, nn.ModuleDict)
            and
            not (module == model)
        ):
            if forward:
                hooks.append(module.register_forward_hook(hook))
            if backward:
                hooks.append(module.register_backward_hook(hook))

    return register_hook, hooks


def get_activations(model, input):

    activations = {}

    def store_activations(module, input, output):
        if isinstance(module, nn.ReLU):
            # TODO ResNet18 implementation reuses a
            # single ReLU layer?
            return
        assert module not in activations, \
            f"{module} already in activations"
        # TODO [0] means first input, not all models have a single input
        activations[module] = (input[0].detach().cpu().numpy().copy(),
                               output.detach().cpu().numpy().copy(),)

    fn, hooks = hook_applyfn(store_activations, model, forward=True)
    model.apply(fn)
    with torch.no_grad():
        model(input)

    for h in hooks:
        h.remove()

    return activations


def memory_size(model, input, as_bits=False):
    """Compute memory size estimate

    Note that this is computed for training purposes, since
    all input activations to parametric are accounted for.

    For inference time you can free memory as you go but with
    residual connections you are forced to remember some, thus
    this is left to implement (TODO)
    The input is required in order to materialize activations
    for dimension independent layers. E.g. Conv layers work

    for any height width.

    Arguments:
        model {torch.nn.Module} -- [description]
        input {torch.Tensor} --

    Keyword Arguments:
        as_bits {bool} -- [description] (default: {False})

    Returns:
        tuple:
         - int -- Estimated memory needed for the full model
         - int -- Estimated memory needed for nonzero activations
    """

    batch_size = input.size(0)
    total_memory = np.prod(input.shape)

    activations = get_activations(model, input)

    # TODO only count parametric layers
    # Input activations are the ones we need for backprop
    input_activations = [i for _, (i, o) in activations.items()]
    for act in input_activations:
        t = np.prod(act.shape)
        if as_bits:
            bits = dtype2bits[act.dtype]
            t *= bits
        total_memory += np.prod(act.shape)

    return total_memory/batch_size
