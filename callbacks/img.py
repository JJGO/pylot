import base64
import io
import json
import pathlib
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any

import jinja2
from box import Box
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pylot.util.img import toImg, torch_renorm
from pylot.nn.hooks import HookedModule


def get_templateEnv() -> jinja2.Environment:
    path = pathlib.Path(__file__).parent / "templates"
    templateLoader = jinja2.FileSystemLoader(searchpath=path)
    templateEnv = jinja2.Environment(
        loader=templateLoader, undefined=jinja2.StrictUndefined
    )
    return templateEnv


def tensor2html(tensor: Tensor, zoom: Optional[int] = None) -> str:

    pil_image = toImg(torch_renorm(tensor))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    b64_image = base64.b64encode(buffered.getvalue())

    width = ""
    if zoom:
        width = f" width: {pil_image.width * zoom}px"

    html_img = (
        f'<img style="{width}" src="data:image/png;base64,{b64_image.decode()}" />'
    )

    return html_img


def sample_batches(
    dataloader: DataLoader, batches: int
) -> Tuple[List[Tensor], List[Tensor]]:
    X, Y = [], []
    it = iter(dataloader)
    for _ in range(batches):
        x, y = next(it)
        X.append(x)
        Y.append(y)
    return X, Y


class TemplateCallback:

    _template = None

    def __init__(self, name):
        self.name = name if name else self._template
        self.output_path = self.experiment.path / f"images/{self.name}.html"
        self.data_path = self.experiment.path / f"images/{self.name}.json.gz"
        self.output_path.parent.mkdir(exist_ok=True, parents=True)

        self.data = Box()
        self.templateEnv = get_templateEnv()

    def __call__(self, epoch):
        self.update(epoch)
        template = self.templateEnv.get_template(self._template + ".j2")
        self.data.epochs = epoch + 1
        html = template.render(**self.data.to_dict())
        with open(self.output_path, "w") as f:
            print(html, file=f)

        if self.data_path.exists():
            os.remove(self.data_path)
            with open(f"{self.name}.json", "w") as f:
                json.dump(self.data.to_dict(), f)

        return html

    def update(self, epoch):
        raise NotImplementedError


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
