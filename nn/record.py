import numpy as np
import pandas as pd
import torch
from torch import nn

from contextlib import contextmanager

from pylot.nn.hooks import HookedModule
from pylot.util.meta import store_attr
from pylot.nn.hyper import VoidModule

from fnmatch import fnmatch


def dict_to_df(dict_):
    size = sum(p.numel() for p in dict_.values())
    names = np.empty((size,), dtype=object)
    values = np.empty((size,), dtype=np.float32)

    i = 0
    for name, weight in dict_.items():
        w = weight.detach().cpu().numpy().flatten()
        N = len(w)
        values[i : i + N] = w
        names[i : i + N] = name
        i += N

    return pd.DataFrame.from_dict({"name": names, "value": values})


def as_flat(dict_):
    return torch.cat([x.flatten() for x in dict_.values()])


#
class Recorder:

    DEFAULT_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)

    def __init__(
        self,
        module,
        module_types=None,
        glob=None,
    ):
        self.module = module
        self.module_types = module_types or self.DEFAULT_MODULES
        self.glob = glob

        self.tracked_modules = set()
        for name, module in self.module.named_modules():
            if not isinstance(module, self.module_types):
                continue
            if glob and not fnmatch(name, glob):
                continue
            self.tracked_modules.add(name)

    @contextmanager
    def record(self, record=True):
        if not record:
            yield
        else:
            module_names = {m: k for k, m in self.module.named_modules()}
            activations = {}

            def _save(module, input, outputs):
                name = module_names[module]
                assert name not in activations
                activations[name] = outputs

            with HookedModule(
                self.module, _save, module_types=self.module_types, glob=self.glob
            ):
                yield self.module

            self.activations = activations

    def weights(self):
        return {
            m_name + "." + p_name: param
            for m_name, module in self.module.named_modules()
            for p_name, param in module._parameters.items()
            if m_name in self.tracked_modules
        }

    def grads(self):
        return {k: w.grad for k, w in self.weights().items()}

    def weights_df(self):
        df = dict_to_df(self.weights())
        name = df.pop("name").str.rpartition(".")
        df["layer"] = name[0]
        df["param"] = name[2]
        df.rename(columns={"value": "weight"}, inplace=True)
        return df[["layer", "param", "weight"]]

    def grads_df(self):
        grads = self.grads()
        df = dict_to_df(grads)
        name = df.pop("name").str.rpartition(".")
        df["layer"] = name[0]
        df["param"] = name[2]
        df.rename(columns={"value": "grad"}, inplace=True)
        return df[["layer", "param", "grad"]]

    def act_df(self):
        adf = dict_to_df(self.activations)
        adf.rename(columns={"name": "layer", "value": "activation"}, inplace=True)
        return adf

    def all(self):
        return {
            "weight": self.weights_df(),
            "grads": self.grads_df(),
            "acts": self.act_df(),
        }

    def grad_stats(self):
        val = as_flat(self.grads())
        return dict(
            grad_mean=val.mean().item(),
            grad_var=val.var().item(),
            grad_norm=val.norm().item(),
        )

    def weight_stats(self):
        val = as_flat(self.weights())
        return dict(
            weight_mean=val.mean().item(),
            weight_var=val.var().item(),
            weight_norm=val.norm().item(),
        )

    def act_stats(self):
        val = as_flat(self.activations)
        return dict(
            act_mean=val.mean().item(),
            act_var=val.var().item(),
            act_norm=val.norm().item(),
        )

    def stats(self):
        return {**self.weight_stats(), **self.grad_stats(), **self.act_stats()}


class VoidRecorder(Recorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.module, VoidModule)

    @contextmanager
    def record(self, record=True):
        if not record:
            yield
        else:
            self._weights = {k: e.data for k, e in self.module.named_externals()}
            module_names = {m: k for k, m in self.module.named_modules()}
            activations = {}

            def _save(module, input, outputs):
                name = module_names[module]
                assert name not in activations
                activations[name] = outputs

            for ext in self.module.externals():
                ext.data.retain_grad()

            with HookedModule(
                self.module, _save, module_types=self.module_types, glob=self.glob
            ):
                yield self.module

            self.activations = activations
            self._grads = {k: e.data.grad for k, e in self.module.named_externals()}

    def weights(self):
        return self._weights

    def grads(self):
        return self._grads


# TODO: default _hook_kw is module_types=(nn.Conv2d, nn.Linear)


# def log_activations(model, *inputs, _hook_kw=None, **kw_inputs):
#     module_names = {m: k for k, m in model.named_modules()}
#     activations = {}

#     def _save(module, input, outputs):
#         name = module_names[module]
#         assert name not in activations
#         activations[name] = outputs

#     with HookedModule(model, _save, **(_hook_kw or {})):
#         y = model(*inputs, **kw_inputs)

#     return y, activations


# def weights_df(model):
#     df = dict_to_df(dict(model.named_parameters()))
#     name = df.pop("name").str.rpartition(".")
#     df["layer"] = name[0]
#     df["weight"] = name[2]
#     df.rename(columns={"value": "weight"}, inplace=True)
#     return df


# def grads_df(model):
#     grads = {k: w.grad for k, w in model.named_parameters()}
#     df = dict_to_df(grads)
#     name = df.pop("name").str.rpartition(".")
#     df["layer"] = name[0]
#     df["weight"] = name[2]
#     df.rename(columns={"value": "grad"}, inplace=True)
#     return df


# def all_dfs(model, input_, y_true, loss_func):

#     y, activations = log_activations(model, input_)

#     loss = loss_func()
#     loss.backward()

#     wdf = weights_df(model)
#     gdf = grads_df(model)
#     adf = dict_to_df(activations)
#     adf.rename(columns={"name": "layer", "value": "activation"}, inplace=True)

#     return {
#         "weight": wdf,
#         "grad": gdf,
#         "act": adf,
#     }
