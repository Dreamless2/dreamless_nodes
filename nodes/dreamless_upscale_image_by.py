import torch
import torch.nn.functional as F

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Upscale_Image_By:    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (
                    ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                    {"default": "bicubic"},
                ),
                "scale_by": (
                    "FLOAT",
                    {"default": 2.00, "min": 0.10, "max": 8.00, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "upscale_by"
    CATEGORY = "Dreamless/Postprocessing"

    def upscale_by(self, image, upscale_method, scale_by):
        target_height = int(image.shape[1] * scale_by)
        target_width = int(image.shape[2] * scale_by)

        print(
            f"{MSG_PREFIX}Scaling image by {scale_by}x to target size: {target_width}x{target_height}..."
        )

        samples = image.permute(0, 3, 1, 2)

        samples = F.interpolate(
            samples,
            size=(target_height, target_width),
            mode=upscale_method,
            align_corners=False if upscale_method != "nearest-exact" else None,
        )

        # Retorna para o formato do ComfyUI [B, H, W, C]
        output_image = samples.permute(0, 2, 3, 1)
        output_image = torch.clamp(output_image, 0.0, 1.0)

        return (output_image,)
