import json
import os
import pathlib

from box import Box

import jinja2


def get_templateEnv() -> jinja2.Environment:
    path = pathlib.Path(__file__).parent / "templates"
    templateLoader = jinja2.FileSystemLoader(searchpath=path)
    templateEnv = jinja2.Environment(
        loader=templateLoader, undefined=jinja2.StrictUndefined
    )
    return templateEnv


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

