from dataclasses import dataclass
from typing import Optional, List, Any

import torch
from torch import Tensor

from ..util.img import toImg, torch_renorm
from ..nn.hooks import HookedModule

from .util import sample_batches, tensor2html
from .template import TemplateCallback


@dataclass
class ImgIO(TemplateCallback):

    experiment: Any
    samples: int = 5
    zoom: Optional[int] = None
    name: Optional[str] = None

    _template = "img_io"

    def __post_init__(self):
        super().__init__(name=self.name)
        X, Y = sample_batches(self.experiment.train_dl, self.samples)
        self.X, self.Y = X, Y

        self.data.samples = self.samples
        self.data.channels_in = self.X[0].shape[1]
        self.data.channels_out = self.Y[0].shape[1]

        self.data.imgX = [tensor2html(x[0], zoom=self.zoom) for x in self.X]
        self.data.imgY = [
            [
                tensor2html(y[0][c : c + 1], zoom=self.zoom)
                for c in range(self.data.channels_out)
            ]
            for y in self.Y
        ]
        self.data.imgYh = {}

    def update(self, epoch):
        self.data.imgYh[epoch] = []
        with torch.no_grad():
            for x in self.X:
                x = x.to(self.experiment.device)
                yh = self.experiment.model(x)

                imgs = [
                    tensor2html(yh[0][c : c + 1], zoom=self.zoom)
                    for c in range(yh.shape[1])
                ]
                self.data.imgYh[epoch].append(imgs)


@dataclass
class ImgActivations(TemplateCallback):

    experiment: Any
    samples: int = 1
    zoom: Optional[int] = None
    modules: Optional[List[str]] = None
    glob: Optional[str] = None
    name: Optional[str] = None

    _template = "img_activations"

    def __post_init__(self):
        super().__init__(name=self.name)
        X, Y = sample_batches(self.experiment.train_dl, self.samples)
        self.X, self.Y = X, Y

        self.data.samples = self.samples
        self.data.channels_in = self.X[0].shape[1]
        self.data.channels_out = self.Y[0].shape[1]

        self.data.imgX = [tensor2html(x[0], zoom=self.zoom) for x in self.X]
        self.data.imgY = [
            [
                tensor2html(y[0][c : c + 1], zoom=self.zoom)
                for c in range(self.data.channels_out)
            ]
            for y in self.Y
        ]
        self.data.activations = {}

        self.modules_to_names = {
            module: name for name, module in self.experiment.model.named_modules()
        }
        self.module_types = (
            tuple(eval(m) for m in self.modules) if self.modules else None
        )

    def update(self, epoch):

        self.data.activations[epoch] = []

        def save_activations(module, _, output):
            if isinstance(output, Tensor):
                activations = [output[0, c : c + 1] for c in range(output.shape[1])]
                imgs = [tensor2html(a, zoom=self.zoom) for a in activations]
                name = self.modules_to_names[module]
                self.data.activations[epoch][-1][name] = imgs

        with torch.no_grad():
            with HookedModule(
                self.experiment.model,
                save_activations,
                module_types=self.module_types,
                glob=self.glob,
            ) as model:
                for x in self.X:
                    self.data.activations[epoch].append({})
                    x = x.to(self.experiment.device)
                    yh = model(x)
                    imgs = [
                        tensor2html(yh[0][c : c + 1], zoom=self.zoom)
                        for c in range(yh.shape[1])
                    ]
                    self.data.activations[epoch][-1]["Output"] = imgs
