import comfy.samplers
import torch
from nodes import KSampler, VAEEncode, VAEDecode, PreviewImage, SaveImage
import folder_paths


class Dreamless_KSampler_Hires:
    @classmethod
    def INPUT_TYPES(cls):
        upscale_models = folder_paths.get_filename_list("upscale_models")
        upscale_models.insert(0, "none")

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
                        "default": 1.00,
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
                "hires_fix": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "hires_upscale": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                    },
                ),
                "hires_steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 10000,
                    },
                ),
                "hires_denoise": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "hires_seed": (["same", "seed+1", "randomize"],),
                "control_after_generation": (
                    ["fixed", "increment", "decrement", "randomize"],
                    {
                        "default": "fixed",
                    },
                ),
                "upscale_model_name": (upscale_models,),
                "upscale_model_override": ("UPSCALE_MODEL",),
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
        control_after_generation,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        image_output,
        filename_prefix,
        latent=None,
        image=None,
        hires_fix=False,
        hires_upscale=2.0,
        hires_steps=8,
        hires_denoise=0.5,
        hires_seed="seed+1",
        upscale_model_name="none",
        upscale_model_override=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        vae_encode = VAEEncode()
        vae_decode = VAEDecode()
        ksampler = KSampler()

        if image is not None:
            latent = vae_encode.encode(vae, image)[0]
        elif latent is None:
            raise ValueError("[Dreamless] Connect a latent or an image!")

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

        if hires_fix:
            upscale_model = self._resolve_upscale_model(
                upscale_model_name,
                upscale_model_override,
            )

            decoded = vae_decode.decode(vae, result_latent)[0]
            b, h, w, c = decoded.shape
            target_h = int(h * hires_upscale)
            target_w = int(w * hires_upscale)

            if upscale_model is not None:
                upscaled = self._upscale_with_model(upscale_model, decoded)
                if upscaled.shape[1] != target_h or upscaled.shape[2] != target_w:
                    upscaled = (
                        torch.nn.functional.interpolate(
                            upscaled.permute(0, 3, 1, 2),
                            size=(target_h, target_w),
                            mode="bicubic",
                            align_corners=False,
                        )
                        .permute(0, 2, 3, 1)
                        .clamp(0, 1)
                    )
            else:
                upscaled = (
                    torch.nn.functional.interpolate(
                        decoded.permute(0, 3, 1, 2),
                        size=(target_h, target_w),
                        mode="bicubic",
                        align_corners=False,
                    )
                    .permute(0, 2, 3, 1)
                    .clamp(0, 1)
                )

            hires_latent = vae_encode.encode(vae, upscaled)[0]
            actual_hires_seed = self._resolve_hires_seed(seed, hires_seed)

            result_latent = ksampler.sample(
                model,
                actual_hires_seed,
                hires_steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                hires_latent,
                denoise=hires_denoise,
            )[0]

        result_image = vae_decode.decode(vae, result_latent)[0]

        next_seed = self._next_seed(seed, control_after_generation)

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

    def _resolve_upscale_model(self, upscale_model_name, upscale_model_override):
        if upscale_model_override is not None:
            return upscale_model_override
        if upscale_model_name and upscale_model_name != "none":
            from nodes import UpscaleModelLoader

            return UpscaleModelLoader().load_model(upscale_model_name)[0]
        return None

    def _upscale_with_model(self, upscale_model, image):
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

        return ImageUpscaleWithModel().upscale(upscale_model, image)[0]

    def _resolve_hires_seed(self, seed, hires_seed):
        if hires_seed == "seed+1":
            return min(seed + 1, 0xFFFFFFFFFFFFFFFF)
        elif hires_seed == "randomize":
            import random

            return random.randint(0, 0xFFFFFFFFFFFFFFFF)
        return seed

    def _next_seed(self, seed, control):
        if control == "increment":
            return min(seed + 1, 0xFFFFFFFFFFFFFFFF)
        elif control == "decrement":
            return max(seed - 1, 0)
        elif control == "randomize":
            import random

            return random.randint(0, 0xFFFFFFFFFFFFFFFF)
        return seed
