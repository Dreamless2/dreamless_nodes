import torch
import torch.nn.functional as F

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Upscale_Image:    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (
                    ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                    {"default": "bicubic"},
                ),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "crop": (["disabled", "center"], {"default": "disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "Dreamless/Postprocessing"

    def upscale(self, image, upscale_method, width, height, crop):
        samples = image.permute(0, 3, 1, 2)

        if crop == "center":
            old_width = samples.shape[3]
            old_height = samples.shape[2]
            old_aspect = old_width / old_height
            new_aspect = width / height

            if old_aspect > new_aspect:
                crop_width = int(old_height * new_aspect)
                crop_x = (old_width - crop_width) // 2
                samples = samples[:, :, :, crop_x : crop_x + crop_width]
            else:
                crop_height = int(old_width / new_aspect)
                crop_y = (old_height - crop_height) // 2
                samples = samples[:, :, crop_y : crop_y + crop_height, :]

        print(
            f"{MSG_PREFIX}Resizing image mathematically to {width}x{height} via {upscale_method}..."
        )

        antialias_modes = {"bilinear", "bicubic", "lanczos"}
        use_antialias = upscale_method in antialias_modes

        align_corners_modes = {"bilinear", "bicubic"}
        use_align_corners = False if upscale_method in align_corners_modes else None

        samples = F.interpolate(
            samples,
            size=(height, width),
            mode=upscale_method,
            align_corners=use_align_corners,
            antialias=use_antialias,
        )

        output_image = samples.permute(0, 2, 3, 1)
        output_image = torch.clamp(output_image, 0.0, 1.0)

        return (output_image,)
