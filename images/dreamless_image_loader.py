import os
import requests
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths
from io import BytesIO
import hashlib

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Image_Loader:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]

        return {
            "required": {
                "mode": (["single", "url"],),
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "urls": ("STRING", {"default": "", "multiline": True}),
                "cache_url": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "COUNT")
    FUNCTION = "load"
    CATEGORY = "Dreamless/Image"

    def load(
        self,
        mode,
        image=None,
        urls="",
        cache_url=True,
    ):
        if mode == "single":
            return self._load_single(image)
        elif mode == "url":
            return self._load_urls(urls, cache_url)

        raise ValueError(f"{MSG_PREFIX}Unknown mode: {mode}")

    def _load_single(self, image_name):
        if not image_name:
            raise ValueError(f"{MSG_PREFIX}No image selected or uploaded!")

        path = folder_paths.get_annotated_filepath(image_name)
        img, mask = self._open_image(path)
        return (img, mask, 1)

    # ------------------------------------------------------------

    def _load_urls(self, urls_text, cache_url):
        if not urls_text or not urls_text.strip():
            raise ValueError(f"{MSG_PREFIX}No URLs provided!")

        # Uma URL por linha, ignora linhas vazias
        url_list = [u.strip() for u in urls_text.splitlines() if u.strip()]

        if not url_list:
            raise ValueError(f"{MSG_PREFIX}No valid URLs found!")

        cache_dir = None
        if cache_url:
            cache_dir = os.path.join(folder_paths.get_input_directory(), "_url_cache")
            os.makedirs(cache_dir, exist_ok=True)

        images = []
        masks = []

        for url in url_list:
            print(f"{MSG_PREFIX}Loading from URL: {url}")
            img, mask = self._load_single_url(url, cache_url, cache_dir)
            images.append(img)
            masks.append(mask)

        images_out = torch.cat(images, dim=0)
        masks_out = torch.cat(masks, dim=0)

        print(f"{MSG_PREFIX}Loaded {len(url_list)} image(s) from URL(s)")

        return (images_out, masks_out, len(url_list))

    # ------------------------------------------------------------

    def _load_single_url(self, url, cache_url, cache_dir):
        if cache_url and cache_dir:
            url_hash = hashlib.md5(url.encode()).hexdigest()
            cache_path = os.path.join(cache_dir, f"{url_hash}.png")

            if os.path.exists(cache_path):
                print(f"{MSG_PREFIX}Cache hit: {cache_path}")
                return self._open_image(cache_path)

        response = requests.get(url, timeout=15)

        if response.status_code != 200:
            raise ValueError(
                f"{MSG_PREFIX}Failed to download image from '{url}'. "
                f"Status: {response.status_code}"
            )

        pil_img = Image.open(BytesIO(response.content))

        if cache_url and cache_dir:
            pil_img.save(cache_path)
            print(f"{MSG_PREFIX}Cached: {cache_path}")

        return self._pil_to_tensor(pil_img)

    def _open_image(self, path):
        pil_img = Image.open(path)
        return self._pil_to_tensor(pil_img)

    def _pil_to_tensor(self, pil_img):
        pil_img = ImageOps.exif_transpose(pil_img)

        rgb = pil_img.convert("RGB")
        img_np = np.array(rgb).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        if pil_img.mode == "RGBA":
            alpha = np.array(pil_img.split()[-1]).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(1.0 - alpha).unsqueeze(0)
        else:
            h, w = img_np.shape[:2]
            mask_tensor = torch.zeros((1, h, w), dtype=torch.float32)

        return img_tensor, mask_tensor
