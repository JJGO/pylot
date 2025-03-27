from ipywidgets import (
    Button,
    Dropdown,
    Checkbox,
    SelectMultiple,
    Text,
    HBox,
    VBox,
    Label,
)
import io
import matplotlib.pyplot as plt
from IPython.display import clear_output, display, HTML

from ..util.jupyter import disable_widget_scroll, widget_dark_mode


class PlotUI:
    def __init__(self, plot_fn, dropdowns, selectors, row_overflow=10):
        self.plot_fn = plot_fn
        self.dropdowns = dropdowns
        self.selectors = selectors
        self.row_overflow = row_overflow
        self.presets = {}
        self.knobs = {}

        # Other attributes
        self.current_fig = None
        self.buttons = None
        self.textbox = None
        self.presets_dd = None
        self.ui = None

        self.build_knobs()
        self.build_ui()

    def knob_values(self):
        kwargs = {k: w.value for k, w in self.knobs.items()}
        key = lambda x: (x.__class__.__name__, x)
        kwargs = {
            k: sorted(v, key=key) if isinstance(v, tuple) else v
            for k, v in kwargs.items()
        }
        return kwargs

    def plot(self):
        kwargs = self.knob_values()
        # self.output.clear_output(wait=True)
        clear_output(wait=True)
        display(self.ui)
        self.current_fig = self.plot_fn(**kwargs)
        f = io.BytesIO()
        plt.savefig(f, format="svg", bbox_inches="tight")
        plt.close()
        self.svg_str = f.getvalue().decode()
        display(HTML(self.svg_str))

    def save_image(self):
        # fig = self.current_fig
        # fig.savefig(f"{self.textbox.value}.pdf", filetype="pdf", bbox_inches="tight")
        with open(f"{self.textbox.value}.svg", "w") as f:
            print(self.svg_str, file=f)

    def save_preset(self):
        self.presets[self.textbox.value] = self.knob_values()
        self.textbox.value = ""
        self.presets_dd.options = list(self.presets.keys())

    def load_preset(self):
        print(self.presets_dd.value)

        preset = self.presets_dd.value
        if isinstance(preset, str):
            preset = self.presets[preset]
        for name, knob in self.knobs.items():
            knob.value = preset[name]

    def build_knobs(self):
        # Dropdowns
        for name, (options, default) in self.dropdowns.items():
            if None not in options:
                options = options + [None]
            self.knobs[name] = Dropdown(
                options=options, description=name, value=default
            )

        # Selectors
        for name, options in self.selectors.items():
            self.knobs[name] = SelectMultiple(
                options=options, value=options, rows=len(options)
            )

        self.knobs["xlog"] = Checkbox(description="xlog", value=False)
        self.knobs["ylog"] = Checkbox(description="ylog", value=False)

    def build_ui_controls(self):

        # Buttons
        self.buttons = {
            "plot": Button(description="Plot"),
            "load_preset": Button(description="Load"),
            "save_preset": Button(description="Save"),
            "save_image": Button(description="Img"),
        }
        self.presets_dd = Dropdown(options=self.presets)
        self.textbox = Text()

        def build_handler(f):
            def handler(_):
                return f()

            return handler

        for name, button in self.buttons.items():
            button.on_click(build_handler(getattr(self, name)))

        ui_controls = HBox(
            [
                self.buttons["plot"],
                self.presets_dd,
                self.buttons["load_preset"],
                self.textbox,
                self.buttons["save_preset"],
                self.buttons["save_image"],
            ]
        )
        return ui_controls

    def build_ui_knobs(self):

        ui_knobs = []
        # First column is Dropdowns (FIXME: use row_overflow for this)
        ui_knobs.append([self.knobs[i] for i in self.dropdowns])
        ui_knobs[-1].extend([self.knobs["xlog"], self.knobs["ylog"]])

        # Make ui for SelectMultiple
        rows = 0
        ui_knobs.append([])
        for name, options in self.selectors.items():
            ui_knobs[-1].append(Label(name))
            ui_knobs[-1].append(self.knobs[name])
            rows += len(options)
            if rows > self.row_overflow:
                rows = 0
                ui_knobs.append([])

        return HBox([VBox(col) for col in ui_knobs])

    def build_ui(self):
        widget_dark_mode()
        disable_widget_scroll()

        ui = VBox(
            [Label("Plot Controls"), self.build_ui_knobs(), self.build_ui_controls()]
        )
        self.ui = ui
        return ui


# def DataFramePlotUI(PlotUI):
#     def __init__(self, data, log_cols, plot_fn, dropdowns, selectors, row_overflow=10):
#         self.data = data
#         self.exp_cols = exp_cols
#         self.log_cols = log_cols
#         dropdowns =
#         selectors = {c: sorted(data[c].unique()) for c in selectors}

#         super().__init__(plot_fn, dropdowns, selectors, row_overflow)
