
import folder_paths
from nodes import (
    UNETLoader,
    DualCLIPLoader,
    VAELoader,
    LoraLoader,
    CLIPTextEncode,
    EmptyLatentImage,
)
from ..utils.helpers import get_api_key, set_api_key, short_paths_map, parse_air
from ..utils.config import BASE_RESOLUTIONS, MAX_RESOLUTION
from ..downloaders.dreamless_downloader import Dreamless_Downloader

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"
RESOLUTION_MAP = {
    f"{w} x {h}": (w, h) for w, h in BASE_RESOLUTIONS if isinstance(w, int)
}

UNET_PATHS = folder_paths.folder_names_and_paths.get("diffusion_models", [["", ""]])[0]
TEXT_ENCODER_PATHS = folder_paths.folder_names_and_paths.get(
    "text_encoders", [["", ""]]
)[0]
VAE_PATHS = folder_paths.folder_names_and_paths.get("vae", [["", ""]])[0]


class Dreamless_Diffusion_Model_DualCLIP_Loader_CivitAI:
    def __init__(self):
        self.unet_loader = None
        self.dual_clip_loader = None
        self.vae_loader = None
        self.lora_loader = None
        self.clip_encoder = None
        self.empty_latent = None

    @classmethod
    def INPUT_TYPES(cls):
        unet_list = folder_paths.get_filename_list("diffusion_models")
        if not unet_list:
            unet_list = ["none"]

        text_encoders = folder_paths.get_filename_list("text_encoders")
        if not text_encoders:
            text_encoders = ["none"]

        vaes = ["none"] + folder_paths.get_filename_list("vae")
        unet_paths = short_paths_map(UNET_PATHS)
        resolution_strings = list(RESOLUTION_MAP.keys()) + ["Custom (width x height)"]

        saved_key = get_api_key("civitai")
        display_key = (
            f"{saved_key[:6]}...{saved_key[-4:]}" if len(saved_key) > 10 else saved_key
        )

        return {
            "required": {
                "unet_air": (
                    "STRING",
                    {"default": "{model_id}@{model_version}", "multiline": False},
                ),
                "unet_name": (["none"] + unet_list,),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16", "fp32"],
                ),
                "clip_name1_air": (
                    "STRING",
                    {"default": "{model_id}@{model_version}", "multiline": False},
                ),
                "clip_name1": (["none"] + text_encoders,),
                "clip_name2_air": (
                    "STRING",
                    {"default": "{model_id}@{model_version}", "multiline": False},
                ),
                "clip_name2": (["none"] + text_encoders,),
                "clip_type": (
                    [
                        "sdxl",
                        "sd3",
                        "flux",
                        "hunyuan_video",
                        "hidream",
                        "hunyuan_image",
                        "hunyuan_video_15",
                        "kandinsky5",
                        "kandinsky5_image",
                        "ltxv",
                        "newbie",
                        "ace",
                    ],
                ),
                "vae_air": (
                    "STRING",
                    {"default": "{model_id}@{model_version}", "multiline": False},
                ),
                "vae_name": (vaes, {"default": "none"}),
                "positive": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Positive Prompt",
                        "dynamicPrompts": True,
                        "height": 300,
                    },
                ),
                "negative": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Negative Prompt",
                        "dynamicPrompts": True,
                        "height": 300,
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
                "unet_download_path": (list(unet_paths.keys()),),
                "optional_model": ("MODEL",),
                "optional_clip": ("CLIP",),
                "optional_vae": ("VAE",),
                "optional_lora_stack": ("LORA_STACK",),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "POSITIVE", "NEGATIVE", "LATENT")
    FUNCTION = "load_model"
    CATEGORY = "Dreamless/Loaders"

    def _resolve_clip(self, air_string, selected_name, token):
        if selected_name != "none":
            return selected_name

        model_id, version_id = parse_air(air_string)
        if not model_id:
            raise ValueError(
                f"{MSG_PREFIX}Please provide a valid AIR for the text encoder or select a local file."
            )

        print(
            f"{MSG_PREFIX}Downloading missing text encoder via Dreamless_Downloader..."
        )
        downloader = Dreamless_Downloader(
            model_id=model_id,
            model_version=version_id,
            model_types=["TextEncoder"],
            token=token,
            save_path=TEXT_ENCODER_PATHS[0],
        )
        if not downloader.download():
            raise RuntimeError(f"{MSG_PREFIX}Failed to download text encoder.")
        return downloader.name

    def load_model(
        self,
        unet_air,
        unet_name,
        weight_dtype,
        clip_name1_air,
        clip_name1,
        clip_name2_air,
        clip_name2,
        clip_type,
        vae_air,
        vae_name,
        positive,
        negative,
        resolution,
        custom_width,
        custom_height,
        batch_size,
        api_key="",
        unet_download_path=None,
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
            if unet_name == "none":
                unet_id, unet_version_id = parse_air(unet_air)
                if not unet_id:
                    raise ValueError(
                        f"{MSG_PREFIX}Please provide a valid diffusion model AIR or select a local file."
                    )

                unet_paths_map = short_paths_map(UNET_PATHS)
                resolved_download_path = (
                    unet_paths_map.get(unet_download_path, UNET_PATHS[0])
                    if unet_download_path
                    else UNET_PATHS[0]
                )

                unet_downloader = Dreamless_Downloader(
                    model_id=unet_id,
                    model_version=unet_version_id,
                    model_types=["Model", "DiffusionModel"],
                    token=token_to_use,
                    save_path=resolved_download_path,
                )
                if not unet_downloader.download():
                    raise RuntimeError(
                        f"{MSG_PREFIX}Failed to download diffusion model."
                    )
                unet_name = unet_downloader.name

                if extra_pnginfo and "workflow" in extra_pnginfo:
                    extra_pnginfo["workflow"].setdefault("extra", {}).setdefault(
                        "unet_airs", []
                    )
                    air = f"{unet_downloader.model_id}@{unet_downloader.version}"
                    if air not in extra_pnginfo["workflow"]["extra"]["unet_airs"]:
                        extra_pnginfo["workflow"]["extra"]["unet_airs"].append(air)
            else:
                print(f"{MSG_PREFIX}Loading diffusion model: {unet_name}")

            if self.unet_loader is None:
                self.unet_loader = UNETLoader()
            model = self.unet_loader.load_unet(
                unet_name=unet_name, weight_dtype=weight_dtype
            )[0]

            final_clip1 = self._resolve_clip(clip_name1_air, clip_name1, token_to_use)
            final_clip2 = self._resolve_clip(clip_name2_air, clip_name2, token_to_use)

            print(
                f"{MSG_PREFIX}Loading dual text encoders setup: {final_clip1} & {final_clip2}"
            )
            if self.dual_clip_loader is None:
                self.dual_clip_loader = DualCLIPLoader()
            clip = self.dual_clip_loader.load_clip(
                clip_name1=final_clip1, clip_name2=final_clip2, type=clip_type
            )[0]

        vae = None
        if optional_vae:
            vae = optional_vae
        else:
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

            if vae_name != "none":
                if self.vae_loader is None:
                    self.vae_loader = VAELoader()
                vae = self.vae_loader.load_vae(vae_name=vae_name)[0]

        if optional_lora_stack:
            if self.lora_loader is None:
                self.lora_loader = LoraLoader()
            for entry in optional_lora_stack:
                if isinstance(entry, (tuple, list)):
                    lora_filename, strength_model, strength_clip = entry
                else:
                    lora_filename = entry.get("name") or entry.get("lora_name")
                    strength_model = entry.get("strength_model", 1.0)
                    strength_clip = entry.get("strength_clip", 1.0)

                if lora_filename and lora_filename != "none":
                    print(
                        f"{MSG_PREFIX}Loading LoRA from stack: {lora_filename} (M:{strength_model}, C:{strength_clip})"
                    )
                    model, clip = self.lora_loader.load_lora(
                        model=model,
                        clip=clip,
                        lora_name=lora_filename,
                        strength_model=strength_model,
                        strength_clip=strength_clip,
                    )

        print(f"{MSG_PREFIX}Encoding text prompts into conditioning vectors...")
        if self.clip_encoder is None:
            self.clip_encoder = CLIPTextEncode()

        positive_conditioning = self.clip_encoder.encode(clip=clip, text=positive)[0]
        negative_conditioning = self.clip_encoder.encode(clip=clip, text=negative)[0]

        if self.empty_latent is None:
            self.empty_latent = EmptyLatentImage()
        latent = self.empty_latent.generate(
            width=width, height=height, batch_size=batch_size
        )[0]

        return (model, clip, vae, positive_conditioning, negative_conditioning, latent)
