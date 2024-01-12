import os
import torch
import requests
from tqdm import tqdm
from diffusers import AutoencoderKL, DiffusionPipeline


def download_file_with_requests_if_not_downloaded(url: str, filename: str) -> str:
    if not os.path.exists(filename):
        response = requests.get(url, stream=True)
        total = int(response.headers.get("content-length", 0))
        with open(filename, "wb") as file, tqdm(
            desc="Downloading model",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                bar.update(size)
    return filename


def defined_model_choice_picker(model_choice: dict, model_dir: str):
    if model_choice["type"] == "diffusers":
        if "vae" in model_choice:
            vae = AutoencoderKL.from_pretrained(
                model_choice["vae"]["url"], torch_dtype=torch.float16
            )
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    model_choice["url"],
                    vae=vae,
                    custom_pipeline="SimpleStableDiffusionXLPipeline",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )
            except ValueError as e:
                pipe = DiffusionPipeline.from_pretrained(
                    model_choice["url"],
                    vae=vae,
                    custom_pipeline="SimpleStableDiffusionXLPipeline",
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                )
        else:
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    model_choice["url"],
                    custom_pipeline="SimpleStableDiffusionXLPipeline",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )
            except ValueError as e:
                pipe = DiffusionPipeline.from_pretrained(
                    model_choice["url"],
                    custom_pipeline="SimpleStableDiffusionXLPipeline",
                    use_safetensors=True,
                )
    elif model_choice["type"] == "civitai":
        civitai_file = download_file_with_requests_if_not_downloaded(
            model_choice["url"], f"{model_dir}/{model_choice['filename']}"
        )
        pipe = DiffusionPipeline.from_pretrained(
            civitai_file,
            vae=vae,
            custom_pipeline="SimpleStableDiffusionXLPipeline",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
    return pipe
