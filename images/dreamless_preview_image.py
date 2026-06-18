

from nodes import PreviewImage
import folder_paths

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Preview_Image:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_dreamless_preview"
        self.preview_node = PreviewImage()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_images"
    OUTPUT_NODE = True
    CATEGORY = "Dreamless/Image"

    def preview_images(self, images, prompt=None, extra_pnginfo=None):
        print(
            f"{MSG_PREFIX}Processing and displaying preview for {images.shape[0]} image(s)..."
        )

        return self.preview_node.save_images(
            images,
            filename_prefix="DreamlessPreview",
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
