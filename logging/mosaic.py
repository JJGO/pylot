import numpy as np
import torch

class TrainValidMosaic:

    def __init__(self, model, x_train, y_train, x_valid, y_valid, x_fn=None):
        self.model = model

        self.x_train = x_train
        self.x_valid = x_valid

        B, C_in, H, W  = x_train.shape
        _, C_out, _, _  = y_train.shape

        self.H_mosaic = B*H
        self.W_mosaic = W*6
        self.C_mosaic = max(C_in, C_out)

        if x_fn is None:
            x_fn = lambda x: x

        self.columns = [
            x_fn(self._to_numpy_img(x_train)),
            None,
            self._to_numpy_img(y_train),
            x_fn(self._to_numpy_img(x_valid)),
            None,
            self._to_numpy_img(y_valid),
        ]

    def _to_numpy_img(self, x):
        x = x.detach().cpu().numpy()
        if x.shape[1] < self.C_mosaic:
            x = x.repeat(self.C_mosaic, axis=1)
        x = x.transpose((0,2,3,1))
        return x.reshape(self.H_mosaic, -1, self.C_mosaic)

    def capture(self):
        with torch.no_grad():
            self.columns[1] = self._to_numpy_img(self.model(self.x_train))
            self.columns[4] = self._to_numpy_img(self.model(self.x_valid))

            img = np.hstack(self.columns)

        return img
