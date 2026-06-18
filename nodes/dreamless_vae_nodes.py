import torch
from nodes import VAEDecode, VAEEncode, VAEDecodeTiled, VAEEncodeTiled

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_VAE_Decode:
    def __init__(self):
        self.decoder = VAEDecode()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Dreamless/VAE"

    def decode(self, samples, vae):
        print(f"{MSG_PREFIX} Running Standard VAE Decode...")
        return self.decoder.decode(samples=samples, vae=vae)


class Dreamless_VAE_Encode:
    def __init__(self):
        self.encoder = VAEEncode()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "Dreamless/VAE"

    def encode(self, pixels, vae):
        print(f"{MSG_PREFIX} Running Standard VAE Encode...")
        return self.encoder.encode(pixels=pixels, vae=vae)


class Dreamless_VAE_Tiled:
    def __init__(self):
        self.decode_tiled = VAEDecodeTiled()
        self.encode_tiled = VAEEncodeTiled()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    [
                        "DECODE (Latent -> Image)",
                        "ENCODE (Image -> Latent)",
                    ],
                    {"default": "DECODE (Latent -> Image)"},
                ),
                "vae": ("VAE",),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 256, "max": 4096, "step": 64},
                ),
            },
            "optional": {
                "samples": ("LATENT",),
                "pixels": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("IMAGE", "LATENT")
    FUNCTION = "process_tiled"
    CATEGORY = "Dreamless/VAE"

    def process_tiled(self, mode, vae, tile_size, samples=None, pixels=None):
        out_image = None
        out_latent = None

        if "DECODE" in mode:
            if samples is None:
                raise ValueError(
                    f"{MSG_PREFIX} Error: DECODE mode selected but 'samples' (LATENT) input is missing."
                )

            print(f"{MSG_PREFIX} Running Tiled VAE Decode (Tile Size: {tile_size})...")
            result = self.decode_tiled.decode(
                samples=samples, vae=vae, tile_size=tile_size
            )
            out_image = result[0]

        else:
            if pixels is None:
                raise ValueError(
                    f"{MSG_PREFIX} Error: ENCODE mode selected but 'pixels' (IMAGE) input is missing."
                )

            print(f"{MSG_PREFIX} Running Tiled VAE Encode (Tile Size: {tile_size})...")
            result = self.encode_tiled.encode(
                pixels=pixels, vae=vae, tile_size=tile_size
            )
            out_latent = result[0]

        if out_image is None:
            out_image = torch.zeros([1, 64, 64, 3])
        if out_latent is None:
            out_latent = {"samples": torch.zeros([1, 4, 8, 8])}

        return (out_image, out_latent)
