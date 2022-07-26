import pathlib
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def summary(model, input_size, batch_size=-1, device="cuda", echo=True, as_stats=False):

    out = ""

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    out += ("----------------------------------------------------------------") + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    out += (line_new) + "\n"
    out += ("================================================================") + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        out += (line_new) + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4.0 / (1024 ** 2.0))
    total_output_size = abs(
        2.0 * total_output * 4.0 / (1024 ** 2.0)
    )  # x2 for gradients
    total_params_size = abs(total_params * 4.0 / (1024 ** 2.0))
    total_size = total_params_size + total_output_size + total_input_size

    out += ("================================================================") + "\n"
    out += ("Total params: {0:,}".format(total_params)) + "\n"
    out += ("Trainable params: {0:,}".format(trainable_params)) + "\n"
    out += (
        "Non-trainable params: {0:,}".format(total_params - trainable_params)
    ) + "\n"
    out += ("----------------------------------------------------------------") + "\n"
    out += ("Input size (MB): %0.2f" % total_input_size) + "\n"
    out += ("Forward/backward pass size (MB): %0.2f" % total_output_size) + "\n"
    out += ("Params size (MB): %0.2f" % total_params_size) + "\n"
    out += ("Estimated Total Size (MB): %0.2f" % total_size) + "\n"
    out += ("----------------------------------------------------------------") + "\n"
    # return summary

    if as_stats:
        return {
            "total_params": int(total_params.item()),
            "trainable_params": int(trainable_params.item()),
            "input_size": int(total_input_size.item()) * 1024 ** 2,
            "backprop_size": int(total_output_size.item()) * 1024 ** 2,
            "params_size": int(total_params_size.item()) * 1024 ** 2,
            "total_size": int(total_size.item()) * 1024 ** 2,
        }

    if echo:
        print(out)
    else:
        return out


def record_model(model, x, path=None):
    if path is None:
        path = "."
    path = pathlib.Path(path)

    # Generate topology
    yhat = model(x)

    from torchviz import make_dot
    g = make_dot(yhat)
    #     g.format = 'svg'
    g.render(path / "topology")

    # Print summary
    with open(path / "summary.txt", "w") as f:
        s = summary(model, x.shape[1:], echo=False)
        print(s, file=f)
