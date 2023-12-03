import random
import PIL
from PIL import ImageFilter
from packaging import version
import requests
import numpy as np
import torch
import os

try:
    from diffusers.utils import PIL_INTERPOLATION
except ImportError:
    if version.parse(version.parse(PIL.__version__).base_version) >= version.parse(
        "9.1.0"
    ):
        PIL_INTERPOLATION = {
            "linear": PIL.Image.Resampling.BILINEAR,
            "bilinear": PIL.Image.Resampling.BILINEAR,
            "bicubic": PIL.Image.Resampling.BICUBIC,
            "lanczos": PIL.Image.Resampling.LANCZOS,
            "nearest": PIL.Image.Resampling.NEAREST,
        }
    else:
        PIL_INTERPOLATION = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
            "nearest": PIL.Image.NEAREST,
        }


def set_seed(seed, display_text=True):
    seed = random.randint(0, 2**32) if seed < 0 else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if display_text:
        print(f"Using the seed {seed}")
    return seed


def load_img(path, shape, add_noise=False):
    if path.startswith("http://") or path.startswith("https://"):
        image = PIL.Image.open(requests.get(path, stream=True).raw).convert("RGB")
    else:
        if os.path.isdir(path):
            files = [
                file
                for file in os.listdir(path)
                if file.endswith(".png") or file.endswith(".jpg")
            ]
            path = os.path.join(path, random.choice(files))
            print(f"Chose random init image {path}")
        image = PIL.Image.open(path).convert("RGB")
    image = image.resize(shape, resample=PIL_INTERPOLATION["lanczos"])
    if add_noise:
        image = image.filter(ImageFilter.GaussianBlur(radius=20))
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    if add_noise:
        noise = np.random.normal(loc=0, scale=1, size=image.shape)
        image = np.clip((image + noise * 0.3), 0, 1)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def free_ram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
