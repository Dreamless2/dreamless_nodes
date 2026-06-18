from nodes import EmptyLatentImage

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"

ASPECT_RATIOS = {
    "1:1 (Square)": 1.0,
    "16:9 (Cinematic Widescreen)": 16.0 / 9.0,
    "9:16 (TikTok / Reels / Shorts)": 9.0 / 16.0,
    "3:2 (Classic Photo)": 3.0 / 2.0,
    "2:3 (Classic Portrait)": 2.0 / 3.0,
    "4:3 (Standard TV / Monitor)": 4.0 / 3.0,
    "3:4 (Tall Photo)": 3.0 / 4.0,
    "21:9 (UltraWide Ultra-Cinematic)": 21.0 / 9.0,
    "1:2 (Ultra Tall Movie Poster)": 1.0 / 2.0,
}


class Dreamless_Smart_Latent:
    def __init__(self):
        self.latent_generator = EmptyLatentImage()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (
                    list(ASPECT_RATIOS.keys()),
                    {"default": "1:1 (Square)"},
                ),
                "orientation": (
                    ["Landscape (Horizontal)", "Portrait (Vertical)"],
                    {"default": "Landscape (Horizontal)"},
                ),
                "target_base": (
                    [
                        "SDXL / Flux (1024px base)",
                        "SD 1.5 (512px base)",
                        "Custom Megapixels",
                    ],
                    {"default": "SDXL / Flux (1024px base)"},
                ),
                "custom_megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("LATENT", "WIDTH", "HEIGHT", "RESOL_STRING")
    FUNCTION = "generate_smart_latent"
    CATEGORY = "Dreamless/Latent"

    def generate_smart_latent(
        self, aspect_ratio, orientation, target_base, custom_megapixels, batch_size
    ):
        if target_base == "SDXL / Flux (1024px base)":
            total_pixels = 1024 * 1024  # 1.05 MP
        elif target_base == "SD 1.5 (512px base)":
            total_pixels = 512 * 512  # 0.26 MP
        else:
            total_pixels = custom_megapixels * 1000000

        ratio = ASPECT_RATIOS[aspect_ratio]

        is_landscape = "Landscape" in orientation
        if (ratio < 1.0 and is_landscape) or (ratio > 1.0 and not is_landscape):
            ratio = 1.0 / ratio

        import math

        height = math.sqrt(total_pixels / ratio)
        width = height * ratio

        final_width = int(round(width / 8.0) * 8)
        final_height = int(round(height / 8.0) * 8)

        resol_str = f"{final_width} x {final_height}"
        print(
            f"{MSG_PREFIX}Smart Latent Activated -> Aspect Ratio: {aspect_ratio} ({orientation}) | Target Resolution: {resol_str}"
        )

        latent = self.latent_generator.generate(
            width=final_width, height=final_height, batch_size=batch_size
        )[0]

        return (latent, final_width, final_height, resol_str)
