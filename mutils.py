import numpy as np
from PIL import Image
from io import BytesIO


def binary2array(binary, dtype):
    """
    Decode the binary as to numpy array of specified data type
    :param binary: bytes,
    :param dtype:
    :return: np.array(dtype)
    """
    return np.array(Image.open(BytesIO(binary)), dtype=dtype)


def array2binary(array: np.ndarray, image_ext='png'):
    """
    Encode the numpy array to binary of specified image format.
    :param array:
    :param image_ext:
    :return:
    """
    bio = BytesIO()
    Image.fromarray(array).save(bio, format=image_ext)
    return bio.getvalue()


def load_colormap(n=256, normalized=False):
    """
    A colormap for displaying mask. Use it like this `plt.imshow(cmap[mask])` and you're
    good to go. Most of the time the default parameters would be ok.

    :param n: int, number of colors in the colormap
    :param normalized: boolean, whether to divide by 255.0 for normalization
    :return:
    """
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((n, 3), dtype=dtype)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap
