import numpy as np
import torch


class TrainValidMosaic:
    def __init__(self, model, x_train, x_valid, y_train=None, y_valid=None, x_fn=None):
        self.model = model

        self.x_train = x_train
        self.x_valid = x_valid

        B, C_in, H, W = x_train.shape
        if y_train is not None:
            _, C_out, _, _ = y_train.shape
        else:
            C_out = C_in

        self.H_mosaic = B * H
        self.W_mosaic = W * 6
        self.C_mosaic = max(C_in, C_out)

        if x_fn is None:
            x_fn = lambda x: x

        self.x_columns = [
            x_fn(self._to_numpy_img(x_train)),
            x_fn(self._to_numpy_img(x_valid)),
        ]

        if y_train is not None and y_valid is not None:

            self.y_columns = [
                self._to_numpy_img(y_train),
                self._to_numpy_img(y_valid),
            ]

        else:
            self.y_columns = [
                np.zeros((self.H_mosaic, 0, self.C_mosaic)),
                np.zeros((self.H_mosaic, 0, self.C_mosaic)),
            ]

    def _to_numpy_img(self, x):
        x = x.detach().cpu().numpy()
        if x.shape[1] < self.C_mosaic:
            x = x.repeat(self.C_mosaic, axis=1)
        x = x.transpose((0, 2, 3, 1))
        return x.reshape(self.H_mosaic, -1, self.C_mosaic)

    def capture(self, output_idx=None):
        with torch.no_grad():
            if output_idx is None:
                yh_columns = [
                    self._to_numpy_img(self.model(self.x_train)),
                    self._to_numpy_img(self.model(self.x_valid)),
                ]
            else:
                yh_columns = [
                    self._to_numpy_img(self.model(self.x_train)[output_idx]),
                    self._to_numpy_img(self.model(self.x_valid)[output_idx]),
                ]

            columns = list(zip(self.x_columns, yh_columns, self.y_columns))
            columns = columns[0] + columns[1]

            img = np.hstack(columns)

        # img = (img*255).astype(np.uint8)
        # if img.shape[-1] == 1:
        #     img = img[...,0]

        return img
