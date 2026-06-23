import comfy.samplers
from nodes import KSampler, VAEDecode

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_KSampler_Config_Advanced:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("SAMPLER_CONFIG",)
    RETURN_NAMES = ("SAMPLER_CONFIG",)
    FUNCTION = "create_config"
    CATEGORY = "Dreamless/Sampling"

    def create_config(self, seed, steps, cfg, sampler_name, scheduler, denoise):
        config = {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
        }
        return (config,)


class Dreamless_KSampler_Standard:
    def __init__(self):
        self.ksampler = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),               
                "sampler_config": ("SAMPLER_CONFIG",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Dreamless/Sampling"

    def sample(self, model, positive, negative, latent_image, sampler_config):
        seed = sampler_config.get("seed", 0)
        steps = sampler_config.get("steps", 20)
        cfg = sampler_config.get("cfg", 7.0)
        sampler_name = sampler_config.get("sampler_name", "euler")
        scheduler = sampler_config.get("scheduler", "normal")
        denoise = sampler_config.get("denoise", 1.0)

        print(
            f"{MSG_PREFIX}Sampling with Config -> Seed: {seed} | Steps: {steps} | CFG: {cfg} | Sampler: {sampler_name} | Scheduler: {scheduler} | Denoise: {denoise}"
        )

        if not self.ksampler:
            self.ksampler = KSampler()

        return self.ksampler.sample(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
        )


class Dreamless_KSampler_Advanced:
    def __init__(self):
        self.ksampler = None
        self.vae_decode = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "sampler_config": ("SAMPLER_CONFIG",),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
        "VAE",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "IMAGE",
    )
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "POSITIVE", "NEGATIVE", "LATENT", "IMAGE")
    FUNCTION = "sample"
    CATEGORY = "Dreamless/Sampling"

    def sample(
        self, model, clip, vae, positive, negative, latent_image, sampler_config
    ):
        seed = sampler_config.get("seed", 0)
        steps = sampler_config.get("steps", 4)
        cfg = sampler_config.get("cfg", 2.0)
        sampler_name = sampler_config.get("sampler_name", "lcm")
        scheduler = sampler_config.get("scheduler", "normal")
        denoise = sampler_config.get("denoise", 1.0)

        print(
            f"{MSG_PREFIX}Sampling Advanced -> Seed: {seed} | Steps: {steps} | CFG: {cfg} | Sampler: {sampler_name} | Scheduler: {scheduler}"
        )

        if not self.ksampler:
            self.ksampler = KSampler()

        sampled_latent = self.ksampler.sample(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
        )[0]
              
        if not self.vae_decode:
            self.vae_decode = VAEDecode()

        image = self.vae_decode.decode(vae=vae, samples=sampled_latent)[0]

        return (model, clip, vae, positive, negative, sampled_latent, image)
