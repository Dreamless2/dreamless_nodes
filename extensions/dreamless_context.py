

import comfy.samplers 

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Context:
    @classmethod
    def INPUT_TYPES(cls):
        samplers_list = comfy.samplers.KSampler.SAMPLERS
        schedulers_list = comfy.samplers.KSampler.SCHEDULERS

        return {
            "required": {},
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "forceInput": True}),
                "steps": ("INT", {"default": 20, "forceInput": True}),
                "cfg": ("FLOAT", {"default": 7.0, "forceInput": True}),
                "sampler": (samplers_list, {"forceInput": True}),
                "scheduler": (schedulers_list, {"forceInput": True}),
                "denoise": ("FLOAT", {"default": 1.0, "forceInput": True}),
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
        "INT",
        "FLOAT",
        comfy.samplers.KSampler.SAMPLERS,
        comfy.samplers.KSampler.SCHEDULERS,
        "FLOAT",
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
        "STEPS",
        "CFG",
        "SAMPLER",
        "SCHEDULER",
        "DENOISE"
    )
    FUNCTION = "process_context"
    CATEGORY = "Dreamless/Context"

    def process_context(self, **kwargs):
        return (
            kwargs.get("model", None),
            kwargs.get("clip", None),
            kwargs.get("vae", None),
            kwargs.get("positive", None),
            kwargs.get("negative", None),
            kwargs.get("latent", None),
            kwargs.get("image", None),
            kwargs.get("seed", 0),
            kwargs.get("steps", 4),
            kwargs.get("cfg", 1.5),
            kwargs.get("sampler", "lcm"),
            kwargs.get("scheduler", "normal"),
            kwargs.get("denoise", 1.0),
        )
