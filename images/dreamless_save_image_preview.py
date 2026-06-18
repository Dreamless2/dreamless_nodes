import os
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Save_Image_Preview:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Dreamless"}),
                "preview": (["ON", "OFF"], {"default": "ON"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Dreamless/Image"

    def save_images(
        self,
        images,
        filename_prefix="Dreamless",
        preview="ON",
        prompt=None,
        extra_pnginfo=None,
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )

        results = list()

        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    metadata.add_text(key, json.dumps(value)) 

            file_name = f"{filename}_{counter:05}_.png"
            file_path = os.path.join(full_output_folder, file_name)

            img.save(file_path, pnginfo=metadata, compress_level=4)

            results.append(
                {
                    "filename": file_name,
                    "subfolder": subfolder,
                    "type": "output",  # <-- sempre output, pois sempre salva
                }
            )
            counter += 1

        print(f"{MSG_PREFIX} Imagens salvas com sucesso em: {full_output_folder}")

        if preview == "ON":
            return {"ui": {"images": results}}
        else:
            return {"ui": {"images": []}}
