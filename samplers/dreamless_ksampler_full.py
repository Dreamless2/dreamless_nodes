import random
import comfy.samplers
from nodes import (
    KSampler,
    VAEEncode,
    VAEDecode,
    PreviewImage,
    SaveImage,
)

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_KSampler_Full:
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 10000,
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": "lcm",
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "default": "sgm_uniform",
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "image_output": (
                    ["Hide", "Preview", "Save", "Hide/Save", "Sender", "Sender/Save"],
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "Dreamless",
                    },
                ),
            },
            "optional": {
                "latent": ("LATENT",),
                "image": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
        "VAE",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "IMAGE",
        "INT",
    )
    RETURN_NAMES = (
        "MODEL",
        "CLIP",
        "VAE",
        "POSITIVE",
        "NEGATIVE",
        "LATENT",
        "IMAGE",
        "SEED",
    )
    FUNCTION = "sample"
    CATEGORY = "Dreamless/Sampling"
    OUTPUT_NODE = True

    def sample(
        self,
        model,
        clip,
        vae,
        positive,
        negative,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        image_output,
        filename_prefix,
        latent=None,
        image=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        vae_encode = VAEEncode()
        vae_decode = VAEDecode()
        ksampler = KSampler()

        if not isinstance(seed, int):
            seed = int(seed)

        if image is not None:
            latent = vae_encode.encode(vae, image)[0]
        elif latent is None:
            raise ValueError(f"{MSG_PREFIX}Connect a latent or an image input!")

        result_latent = ksampler.sample(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]

        result_image = vae_decode.decode(vae, result_latent)[0]
        next_seed = seed

        ui_images = []

        if image_output in ("Preview", "Sender"):
            preview = PreviewImage()
            ui_out = preview.save_images(
                result_image,
                filename_prefix=filename_prefix,
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
            )
            ui_images = ui_out.get("ui", {}).get("images", [])

        elif image_output in ("Save", "Hide/Save", "Sender/Save"):
            saver = SaveImage()
            ui_out = saver.save_images(
                result_image,
                filename_prefix=filename_prefix,
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
            )
            ui_images = ui_out.get("ui", {}).get("images", [])

        return {
            "result": (
                model,
                clip,
                vae,
                positive,
                negative,
                result_latent,
                result_image,
                next_seed,
            ),
            "ui": {"images": ui_images},
        }

    def _next_seed(self, seed, control):
        if control == "increment":
            return min(seed + 1, 0xFFFFFFFFFFFFFFFF)
        elif control == "decrement":
            return max(seed - 1, 0)
        elif control == "randomize":
            return random.randint(0, 0xFFFFFFFFFFFFFFFF)
        return seed  # "fixed"
