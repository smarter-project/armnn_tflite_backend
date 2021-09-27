import numpy as np
from PIL import Image


def image_set_generator(iterable, max_items, min_items=1):
    s = list(iterable)
    return [s[:x] for x in range(min_items, max_items + 1)]


def extract_photo(filename, width, height, scaling=None):
    img = Image.open(filename)
    resized_img = img.resize((width, height), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    expanded = np.expand_dims(resized, axis=0)

    typed = expanded.astype(np.float32)

    if scaling:
        if scaling.lower() == "inception":
            scaled = (typed / 127.5) - 1.0
        elif scaling.lower() == "resnetv2":
            scaled = 2 * (typed / 255.0) - 1.0
        elif scaling.lower() == "mobilenet":
            scaled = (typed / 255.0) - 1.0
        elif scaling.lower() == "mobilenetv3":
            scaled = typed * (1.0 / 255)
        elif scaling.lower() == "ssdmobilenetv1":
            scaled = (2.0 / 255.0) * typed - 1.0
        else:
            scaled = typed
    else:
        scaled = typed

    return scaled
