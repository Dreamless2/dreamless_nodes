import random

import comfy.samplers

from nodes import (
    KSampler,
)

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_KSampler_Simple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
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
                    {"default": "sgm_uniform"},
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
            },
            "optional": {
                "latent": ("LATENT",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("LATENT", "VAE")

    RETURN_NAMES = ("LATENT", "VAE")

    FUNCTION = "sample"
    CATEGORY = "Dreamless/Sampling"
    OUTPUT_NODE = True

    def sample(
        self,
        model,
        vae,
        positive,
        negative,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        latent=None,
        prompt=None,
        extra_pnginfo=None,
    ):

        ksampler = KSampler()

        if latent is None:
            raise ValueError(f"{MSG_PREFIX} No LATENT or IMAGE input provided.")

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

        next_seed = seed

        return (result_latent, vae, next_seed)

    def _next_seed(self, seed, control):
        if control == "increment":
            return min(seed + 1, 0xFFFFFFFFFFFFFFFF)
        elif control == "decrement":
            return max(seed - 1, 0)
        elif control == "randomize":
            return random.randint(0, 0xFFFFFFFFFFFFFFFF)

        return seed
