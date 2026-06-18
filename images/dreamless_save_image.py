import os
import json
import folder_paths

from PIL import Image, PngImagePlugin


class Dreamless_Save_Image:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "Dreamless",
                        "multiline": False,
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Dreamless/IO"

    # =========================================================

    def save_images(
        self,
        images,
        filename_prefix="Dreamless",
        prompt=None,
        extra_pnginfo=None,
    ):

        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            filename_prefix,
        ) = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            images[0].shape[1],
            images[0].shape[0],
        )

        results = []

        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()

            img = Image.fromarray(i.clip(0, 255).astype("uint8"))

            metadata = PngImagePlugin.PngInfo()

            if prompt is not None:
                metadata.add_text(
                    "prompt",
                    json.dumps(prompt),
                )

            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    metadata.add_text(
                        key,
                        json.dumps(value),
                    )

            file = f"{filename}_{counter:05}_.png"

            save_path = os.path.join(
                full_output_folder,
                file,
            )

            img.save(
                save_path,
                pnginfo=metadata,
                compress_level=4,
            )

            results.append(
                {
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type,
                }
            )

            counter += 1

        return {"ui": {"images": results}}
