import folder_paths
from nodes import (
    CheckpointLoaderSimple,
    VAELoader,
    LoraLoader,
    CLIPTextEncode,
    EmptyLatentImage,
)
from ..utils.config import BASE_RESOLUTIONS, MAX_RESOLUTION

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"
RESOLUTION_MAP = {
    f"{w} x {h}": (w, h) for w, h in BASE_RESOLUTIONS if isinstance(w, int)
}


class Dreamless_Loader:
    def __init__(self):
        self.ckpt_loader = CheckpointLoaderSimple()
        self.vae_loader = VAELoader()
        self.lora_loader = LoraLoader()
        self.clip_encoder = CLIPTextEncode()
        self.empty_latent = EmptyLatentImage()

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        checkpoints.insert(0, "none")
        vaes = ["none", "Baked VAE"] + folder_paths.get_filename_list("vae")
        loras = ["none"] + folder_paths.get_filename_list("loras")
        resolution_strings = list(RESOLUTION_MAP.keys()) + ["Custom (width x height)"]

        return {
            "required": {
                "ckpt_name": (checkpoints,),
                "vae_name": (vaes, {"default": "Baked VAE"}),
                "clip_skip": ("INT", {"default": -2, "min": -24, "max": 0, "step": 1}),
                "lora_name": (loras,),
                "lora_strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "lora_strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "positive": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Positive prompt",
                        "dynamicPrompts": True,
                        "min_height": 200,
                    },
                ),
                "negative": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Negative prompt",
                        "min_height": 200,
                        "dynamicPrompts": False,
                    },
                ),
                "resolution": (resolution_strings, {"default": "1024 x 1024"}),
                "custom_width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "custom_height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "optional_model": ("MODEL",),
                "optional_clip": ("CLIP",),
                "optional_vae": ("VAE",),
                "optional_lora_stack": ("LORA_STACK",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "POSITIVE", "NEGATIVE", "LATENT")
    FUNCTION = "load_checkpoint"
    CATEGORY = "Dreamless/Loaders"

    def load_checkpoint(
        self,
        ckpt_name,
        vae_name,
        clip_skip,
        lora_name,
        lora_strength_model,
        lora_strength_clip,
        positive,
        negative,
        resolution,
        custom_width,
        custom_height,
        batch_size,
        optional_model=None,
        optional_clip=None,
        optional_vae=None,
        optional_lora_stack=None,
    ):
        if resolution == "Custom (width x height)":
            width, height = custom_width, custom_height
        else:
            width, height = RESOLUTION_MAP[resolution]

        if optional_model and optional_clip:
            model = optional_model
            clip = optional_clip
            vae = optional_vae
        else:
            if ckpt_name == "none":
                raise ValueError(f"{MSG_PREFIX}Please select a valid checkpoint.")
            print(f"{MSG_PREFIX}Loading checkpoint: {ckpt_name}")
            model, clip, vae = self.ckpt_loader.load_checkpoint(ckpt_name=ckpt_name)

        if optional_vae:
            vae = optional_vae
        elif vae_name != "Baked VAE" and vae_name != "none":
            vae = self.vae_loader.load_vae(vae_name=vae_name)[0]

        if lora_name != "none":
            model, clip = self.lora_loader.load_lora(
                model=model,
                clip=clip,
                lora_name=lora_name,
                strength_model=lora_strength_model,
                strength_clip=lora_strength_clip,
            )

        if optional_lora_stack:
            for entry in optional_lora_stack:
                if isinstance(entry, (tuple, list)):
                    lora_name, strength_model, strength_clip = entry
                else:
                    lora_name = entry.get("name") or entry.get("lora_name")
                    strength_model = entry.get("strength_model", 1.0)
                    strength_clip = entry.get("strength_clip", 1.0)

                if lora_name and lora_name != "none":
                    print(f"\33[1m\33[36m[Dreamless] Load LoRA: \33[0m{lora_name}: model: {strength_model}, clip:{strength_clip}")
                    model, clip = self.lora_loader.load_lora(
                        model=model,
                        clip=clip,
                        lora_name=lora_name,
                        strength_model=strength_model,
                        strength_clip=strength_clip,
                    )

        if clip_skip != -1:
            clip = clip.clone()
            clip.clip_layer(clip_skip)

        positive_conditioning = self.clip_encoder.encode(clip=clip, text=positive)[0]
        negative_conditioning = self.clip_encoder.encode(clip=clip, text=negative)[0]
        latent = self.empty_latent.generate(
            width=width, height=height, batch_size=batch_size
        )[0]

        return (model, clip, vae, positive_conditioning, negative_conditioning, latent)
