

import comfy.samplers


MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_KSampler_Config_Simple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
                        "step": 1,
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
                    {"default": "lcm"},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"default": "normal"},
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
            }
        }

    RETURN_TYPES = (
        "INT",
        "INT",
        "FLOAT",
        comfy.samplers.KSampler.SAMPLERS,
        comfy.samplers.KSampler.SCHEDULERS,
        "FLOAT",
    )
    RETURN_NAMES = ("SEED", "STEPS", "CFG", "SAMPLER", "SCHEDULER", "DENOISE")

    FUNCTION = "get_config"
    CATEGORY = "Dreamless/Sampling"

    def get_config(self, seed, steps, cfg, sampler_name, scheduler, denoise):
        next_seed = seed
        return (next_seed, steps, cfg, sampler_name, scheduler, denoise)

    def _next_seed(self, seed, control):
        if control == "increment":
            return min(seed + 1, 0xFFFFFFFFFFFFFFFF)
        elif control == "decrement":
            return max(seed - 1, 0)
        elif control == "randomize":
            import random

            return random.randint(0, 0xFFFFFFFFFFFFFFFF)
        return seed
