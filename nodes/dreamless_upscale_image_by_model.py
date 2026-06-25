import torch
import folder_paths
import torch.nn.functional as F

from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Upscale_Image_By_Model:
    def __init__(self):
        self.model_loader = UpscaleModelLoader()
        self.upscaler = ImageUpscaleWithModel()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": (folder_paths.get_filename_list("upscale_models"),),
                "upscale_method": (
                    ["nearest_exact", "bilinear", "area", "bicubic", "lanczos"],
                    {"default": "bicubic"},
                ),
                "scale_by": (
                    "FLOAT",
                    {"default": 2.00, "min": 0.10, "max": 8.00, "step": 0.01},
                ),
                "tile_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2048,
                        "step": 32,
                        "tooltip": "Tile size for tiled upscale. 0 = disabled (full image).",
                    },
                ),
                "tile_overlap": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 256,
                        "step": 8,
                        "tooltip": "Overlap between tiles in pixels to avoid seams.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "upscale_by"
    CATEGORY = "Dreamless/Postprocessing"

    def upscale_by(
        self, image, upscale_model, upscale_method, scale_by, tile_size, tile_overlap
    ):
        print(f"{MSG_PREFIX}Loading upscale model: {upscale_model}")
        model_loaded = self.model_loader.load_model(upscale_model)[0]

        target_height = int(image.shape[1] * scale_by)
        target_width = int(image.shape[2] * scale_by)

        print(
            f"{MSG_PREFIX}Target dimensions: {target_width}x{target_height} ({scale_by}x)"
        )

        if tile_size > 0:
            print(
                f"{MSG_PREFIX}Using tiled upscale (tile={tile_size}, overlap={tile_overlap})..."
            )
            model_upscaled = self._upscale_tiled(
                model_loaded, image, tile_size, tile_overlap
            )
        else:
            print(f"{MSG_PREFIX}Applying full image model upscale...")
            model_upscaled = self.upscaler.upscale(model_loaded, image)[0]

        samples = model_upscaled.permute(0, 3, 1, 2)

        print(f"{MSG_PREFIX}Resizing to final factor via {upscale_method}...")
        
        antialias_modes = {"bilinear", "bicubic", "lanczos"}
        use_antialias = upscale_method in antialias_modes

        align_corners_modes = {"bilinear", "bicubic"}
        use_align_corners = False if upscale_method in align_corners_modes else None

        samples = F.interpolate(
            samples,
            size=(target_height, target_width),
            mode=upscale_method,
            align_corners=use_align_corners,
            antialias=use_antialias,
        )

        output_image = samples.permute(0, 2, 3, 1)
        output_image = torch.clamp(output_image, 0.0, 1.0)

        return (output_image,)


    def _upscale_tiled(self, model, image, tile_size, overlap):
        b, h, w, c = image.shape
        scale = self._detect_model_scale(model, image)

        out_h = h * scale
        out_w = w * scale
        output = torch.zeros(
            (b, out_h, out_w, c), dtype=image.dtype, device=image.device
        )
        weight = torch.zeros(
            (b, out_h, out_w, c), dtype=image.dtype, device=image.device
        )

        step = max(tile_size - overlap, 1)

        y_starts = list(range(0, h, step))
        x_starts = list(range(0, w, step))

        total = len(y_starts) * len(x_starts)
        current = 0

        for y in y_starts:
            for x in x_starts:
                current += 1
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                y0 = max(y_end - tile_size, 0)
                x0 = max(x_end - tile_size, 0)

                tile = image[:, y0:y_end, x0:x_end, :]

                print(
                    f"\r{MSG_PREFIX}Tile {current}/{total} | "
                    f"[{x0}:{x_end}, {y0}:{y_end}]",
                    end="",
                    flush=True,
                )

                upscaled_tile = self.upscaler.upscale(model, tile)[0]

                oy0, oy1 = y0 * scale, y_end * scale
                ox0, ox1 = x0 * scale, x_end * scale

                output[:, oy0:oy1, ox0:ox1, :] += upscaled_tile
                weight[:, oy0:oy1, ox0:ox1, :] += 1.0

        print()
        output = output / weight.clamp(min=1e-6)
        return output.clamp(0.0, 1.0)

    def _detect_model_scale(self, model, image):
        small = image[:1, :16, :16, :]
        with torch.no_grad():
            out = self.upscaler.upscale(model, small)[0]
        return out.shape[1] // 16
