

import folder_paths

from nodes import (
    CheckpointLoaderSimple,
    VAELoader,
    LoraLoader,
    CLIPTextEncode,
    EmptyLatentImage,
)

from ..utils.helpers import (
    get_api_key,
    set_api_key,
    short_paths_map,
    parse_air,
)

from ..utils.config import BASE_RESOLUTIONS, MAX_RESOLUTION
from ..downloaders.dreamless_downloader import Dreamless_Downloader

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"
RESOLUTION_MAP = {
    f"{w} x {h}": (w, h) for w, h in BASE_RESOLUTIONS if isinstance(w, int)
}

CHECKPOINTS = folder_paths.folder_names_and_paths["checkpoints"][0]
LORAS = folder_paths.folder_names_and_paths.get("loras", [["", ""]])[0]
VAE_PATHS = folder_paths.folder_names_and_paths.get("vae", [["", ""]])[0]

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Loader_CivitAI:
    def __init__(self):
        self.ckpt_loader = CheckpointLoaderSimple()
        self.vae_loader = VAELoader()
        self.lora_loader = LoraLoader()
        self.clip_encoder = CLIPTextEncode()
        self.empty_latent = EmptyLatentImage()

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = ["none"] + folder_paths.get_filename_list("checkpoints")
        loras = ["none"] + folder_paths.get_filename_list("loras")
        vaes = ["none", "Baked VAE"] + folder_paths.get_filename_list("vae")
        checkpoint_paths = short_paths_map(CHECKPOINTS)
        resolution_strings = list(RESOLUTION_MAP.keys()) + ["Custom (width x height)"]

        saved_key = get_api_key("civitai")
        display_key = (
            f"{saved_key[:6]}...{saved_key[-4:]}" if len(saved_key) > 10 else saved_key
        )

        return {
            "required": {
                "ckpt_air": (
                    "STRING",
                    {"default": "{model_id}@{model_version}", "multiline": False},
                ),
                "ckpt_name": (checkpoints,),
                "vae_air": (
                    "STRING",
                    {"default": "{model_id}@{model_version}", "multiline": False},
                ),
                "vae_name": (vaes, {"default": "Baked VAE"}),
                "clip_skip": ("INT", {"default": -2, "min": -24, "max": -1, "step": 1}),
                "lora_air": (
                    "STRING",
                    {"default": "{model_id}@{model_version}", "multiline": False},
                ),
                "lora_name": (loras, {"default": "none"}),
                "lora_strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
                "lora_strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
                "positive": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Positive Prompt",
                        "dynamicPrompts": True,
                        "min_height": 200,
                    },
                ),
                "negative": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Negative Prompt",
                        "dynamicPrompts": True,
                        "min_height": 200,
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
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "api_key": ("STRING", {"default": display_key, "multiline": False}),
                "download_path": (list(checkpoint_paths.keys()),),
                "optional_model": ("MODEL",),
                "optional_clip": ("CLIP",),
                "optional_vae": ("VAE",),
                "optional_lora_stack": ("LORA_STACK",),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "POSITIVE", "NEGATIVE", "LATENT")
    FUNCTION = "load_checkpoint"
    CATEGORY = "Dreamless/Loaders"

    def load_checkpoint(
        self,
        ckpt_air,
        ckpt_name,
        vae_air,
        vae_name,
        clip_skip,
        positive,
        negative,
        resolution,
        custom_width,
        custom_height,
        batch_size,
        lora_air="none",
        lora_name="none",
        lora_strength_model=1.0,
        lora_strength_clip=1.0,
        api_key="",
        download_path=None,
        optional_model=None,
        optional_clip=None,
        optional_vae=None,
        optional_lora_stack=None,
        extra_pnginfo=None,
    ):

        saved_key = get_api_key("civitai")
        if (
            api_key
            and api_key.strip()
            and not api_key.startswith(saved_key[:6] + "...")
        ):
            set_api_key("civitai", api_key)
            token_to_use = api_key.strip()
        else:
            token_to_use = saved_key

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
                ckpt_id, version_id = parse_air(ckpt_air)

                if not ckpt_id:
                    raise ValueError(f"{MSG_PREFIX} Invalid URL.")

                checkpoint_paths = short_paths_map(CHECKPOINTS)
                if download_path and download_path in checkpoint_paths:
                    download_path = checkpoint_paths[download_path]
                else:
                    download_path = CHECKPOINTS[0]

                model_downloader = Dreamless_Downloader(
                    model_id=ckpt_id,
                    model_version=version_id,
                    model_types=["Checkpoint"],
                    token=token_to_use,
                    save_path=download_path,
                )

                if not model_downloader.download():
                    raise RuntimeError(f"{MSG_PREFIX} Failed to download checkpoint.")

                ckpt_name = model_downloader.name

                if extra_pnginfo and "workflow" in extra_pnginfo:
                    extra_pnginfo["workflow"].setdefault("extra", {}).setdefault(
                        "ckpt_airs", []
                    )
                    air = f"{model_downloader.model_id}@{model_downloader.version}"
                    if air not in extra_pnginfo["workflow"]["extra"]["ckpt_airs"]:
                        extra_pnginfo["workflow"]["extra"]["ckpt_airs"].append(air)
            else:
                print(f"{MSG_PREFIX}Loading checkpoint: {ckpt_name}")

            model, clip, vae = self.ckpt_loader.load_checkpoint(ckpt_name=ckpt_name)

        if vae_name == "none" and vae_air and vae_air.strip() != "none":
            vae_id, vae_version_id = parse_air(vae_air)

            if vae_id:
                vae_civitai = Dreamless_Downloader(
                    model_id=vae_id,
                    model_version=vae_version_id,
                    model_types=["VAE"],
                    token=token_to_use,
                    save_path=VAE_PATHS[0],
                )

                if vae_civitai.download():
                    vae_name = vae_civitai.name

        if optional_vae:
            vae = optional_vae
        elif vae_name != "Baked VAE" and vae_name != "none":
            vae = self.vae_loader.load_vae(vae_name=vae_name)[0]

        if lora_name == "none" and lora_air and lora_air.strip() != "none":
            lora_id, lora_version_id = parse_air(lora_air)
            if lora_id:
                lora_civitai = Dreamless_Downloader(
                    model_id=lora_id,
                    model_version=lora_version_id,
                    model_types=["Lora", "LyCORIS"],
                    token=token_to_use,
                    save_path=LORAS[0],
                )
                if lora_civitai.download():
                    lora_name = lora_civitai.name

        if lora_name and lora_name != "none":
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
                    print(f"\33[1m\33[36m[Dreamless] Load LoRA: \33[0m{lora_name}: Model {strength_model} | Clip {strength_clip}")
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