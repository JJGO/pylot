import base64
import io
from typing import List, Tuple, Optional

from torch import Tensor
from torch.utils.data import DataLoader

from ..util.img import toImg


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


def tensor2html(tensor: Tensor, zoom: Optional[int] = None, norm: bool = True) -> str:

    pil_image = toImg(tensor, norm=norm)
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
