import random
import numpy as np
import cv2

# Ellipse toy data generators
def gen_random_image(shape):
    H, W, C = shape
    img = np.zeros((H, W, C), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)

    # Background
    dark_color0 = random.randint(0, 100)
    dark_color1 = random.randint(0, 100)
    dark_color2 = random.randint(0, 100)
    img[:, :, 0] = dark_color0
    img[:, :, 1] = dark_color1
    img[:, :, 2] = dark_color2

    # Object
    light_color0 = random.randint(dark_color0+1, 255)
    light_color1 = random.randint(dark_color1+1, 255)
    light_color2 = random.randint(dark_color2+1, 255)
    center_0 = random.randint(0, H)
    center_1 = random.randint(0, W)
    r1 = random.randint(10, 56)
    r2 = random.randint(10, 56)
    cv2.ellipse(img, (center_0, center_1), (r1, r2), 0, 0, 360, (light_color0, light_color1, light_color2), -1)
    cv2.ellipse(mask, (center_0, center_1), (r1, r2), 0, 0, 360, 255, -1)

    # White noise
    density = random.uniform(0, 0.1)
    for i in range(H):
        for j in range(W):
            if random.random() < density:
                img[i, j, 0] = random.randint(0, 255)
                img[i, j, 1] = random.randint(0, 255)
                img[i, j, 2] = random.randint(0, 255)

    return img, mask

def EllipseGenerator(batch_size, shape):
    while True:
        image_list = []
        mask_list = []
        for i in range(batch_size):
            img, mask = gen_random_image(shape)
            image_list.append(img)
            mask_list.append(mask)

        image_list = np.array(image_list, dtype=np.float64)
        image_list /= 255.0
        image_list -= 0.5
        mask_list = np.array(mask_list, dtype=np.float64)
        mask_list /= 255.0
        mask_list = mask_list[..., np.newaxis]
        yield image_list, mask_list


from torch.utils.data import Dataset

class EllipseSegmentation(Dataset):


    def __init__(self, shape, epoch=1, normalize=True):
        if isinstance(shape, int):
            self.shape = (shape, shape, 3)
        elif isinstance(shape, tuple):
            self.shape = (*shape[:2], 3)

        self.epoch = epoch
        self.normalize = normalize

    def __len__(self):
        return self.epoch

    def __getitem__(self, idx):
        img, mask = gen_random_image(self.shape)
        img = np.array(img, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)[..., np.newaxis]

        if self.normalize:
            img = (img / 255.0) - 0.5
            mask /= 255.0

        # Torch wants channels first
        img = img.transpose((2,0,1))
        mask = mask.transpose((2,0,1))

        return img, mask