from .mosaic import TrainValidMosaic




import pathlib
from torchviz import make_dot
from ..util.summary import summary


def record_model(model, x, path=None):
    if path is None:
        path = '.'
    path = pathlib.Path(path)

    # Generate topology
    yhat = model(x)
    g = make_dot(yhat)
#     g.format = 'svg'
    g.render(path / 'topology')

    # Print summary
    with open( path / 'summary.txt', 'w') as f:
        s = summary(model, x.shape[1:], echo=False)
        print(s, file=f)
