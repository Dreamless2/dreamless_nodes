import torch

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Image_Adjustments:
    COLOR = "#1b365d"
    BG_COLOR = "#0f172a"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},
                ),
                "sharpness": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 5.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "apply_adjustments"
    CATEGORY = "Dreamless/Postprocessing"

    def apply_adjustments(self, image, brightness, contrast, saturation, sharpness):
        print(f"{MSG_PREFIX}Applying image adjustments...")

        # O ComfyUI trabalha com tensores no formato [B, H, W, C] clonados
        output_image = image.clone()

        # =====================================================
        # 1. BRIGHTNESS
        # =====================================================
        if brightness != 1.0:
            output_image = output_image * brightness

        # =====================================================
        # 2. CONTRAST
        # =====================================================
        if contrast != 1.0:
            # Usa a média cinza global como ponto pivô do contraste
            mean = torch.mean(output_image, dim=(1, 2), keepdim=True)
            output_image = (output_image - mean) * contrast + mean

        # =====================================================
        # 3. SATURATION
        # =====================================================
        if saturation != 1.0:
            grayscale = (
                output_image[..., 0] * 0.299
                + output_image[..., 1] * 0.587
                + output_image[..., 2] * 0.114
            )
            grayscale = grayscale.unsqueeze(-1)
            output_image = torch.lerp(grayscale, output_image, saturation)

        # =====================================================
        # 4. SHARPNESS (Sharpen / Blur)
        # =====================================================
        if sharpness != 0.0:
            samples = output_image.permute(0, 3, 1, 2)

            padding = (1, 1, 1, 1)
            padded = torch.nn.functional.pad(samples, padding, mode="reflect")

            blurred = (
                padded[:, :, 0:-2, 0:-2]
                + padded[:, :, 0:-2, 1:-1]
                + padded[:, :, 0:-2, 2:]
                + padded[:, :, 1:-1, 0:-2]
                + padded[:, :, 1:-1, 1:-1]
                + padded[:, :, 1:-1, 2:]
                + padded[:, :, 2:, 0:-2]
                + padded[:, :, 2:, 1:-1]
                + padded[:, :, 2:, 2:]
            ) / 9.0

            high_pass = samples - blurred
            samples = samples + (high_pass * sharpness)

            output_image = samples.permute(0, 2, 3, 1)

        # =====================================================
        # CRITICAL PROTECTION
        # =====================================================
        output_image = torch.clamp(output_image, 0.0, 1.0)

        print(f"{MSG_PREFIX}Adjustments applied successfully.")
        return (output_image,)
