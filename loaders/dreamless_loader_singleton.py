import folder_paths
import os

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
    is_hf,
    parse_hf,
)

from ..utils.config import BASE_RESOLUTIONS, MAX_RESOLUTION
from ..downloaders.dreamless_downloader import Dreamless_Downloader
from ..downloaders.dreamless_hf_downloader import Dreamless_HF_Downloader

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"
RESOLUTION_MAP = {
    f"{w} x {h}": (w, h) for w, h in BASE_RESOLUTIONS if isinstance(w, int)
}

CHECKPOINTS = folder_paths.folder_names_and_paths["checkpoints"][0]
LORAS = folder_paths.folder_names_and_paths.get("loras", [["", ""]])[0]
VAE_PATHS = folder_paths.folder_names_and_paths.get("vae", [["", ""]])[0]


class Dreamless_Loader_Singleton:
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

        saved_civitai = get_api_key("civitai")
        display_civitai = (
            f"{saved_civitai[:6]}...{saved_civitai[-4:]}"
            if len(saved_civitai) > 10
            else saved_civitai
        )

        saved_hf = get_api_key("huggingface")
        display_hf = (
            f"{saved_hf[:6]}...{saved_hf[-4:]}" if len(saved_hf) > 10 else saved_hf
        )

        return {
            "required": {
                "ckpt_air": (
                    "STRING",
                    {
                        "default": "{model_id}@{model_version}",
                        "multiline": False,
                        "tooltip": "CivitAI: {model_id}@{version} | HF: owner/repo/file.safetensors[@branch] | hf:owner/repo/file.safetensors",
                    },
                ),
                "ckpt_name": (checkpoints,),
                "vae_air": (
                    "STRING",
                    {
                        "default": "{model_id}@{model_version}",
                        "multiline": False,
                        "tooltip": "CivitAI: {model_id}@{version} | HF: owner/repo/file.safetensors[@branch]",
                    },
                ),
                "vae_name": (vaes, {"default": "Baked VAE"}),
                "clip_skip": ("INT", {"default": -2, "min": -24, "max": -1, "step": 1}),
                "lora_air": (
                    "STRING",
                    {
                        "default": "{model_id}@{model_version}",
                        "multiline": False,
                        "tooltip": "CivitAI: {model_id}@{version} | HF: owner/repo/file.safetensors[@branch]",
                    },
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
                "civitai_api_key": (
                    "STRING",
                    {"default": display_civitai, "multiline": False},
                ),
                "hf_api_key": (
                    "STRING",
                    {"default": display_hf, "multiline": False},
                ),
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

    def _resolve_token(self, input_key: str, service: str) -> str:
        saved = get_api_key(service)
        if (
            input_key
            and input_key.strip()
            and not input_key.startswith(saved[:6] + "...")
        ):
            set_api_key(service, input_key.strip())
            return input_key.strip()
        return saved

    def _download_ckpt(
        self, air: str, civitai_token: str, hf_token: str, download_path: str
    ) -> str:
        if is_hf(air):
            repo_id, filename, revision = parse_hf(air)
            if not repo_id or not filename:
                raise ValueError(f"{MSG_PREFIX} HF reference inválida: {air}")
            dl = Dreamless_HF_Downloader(
                repo_id=repo_id,
                filename=filename,
                token=hf_token,
                revision=revision,
                save_path=download_path,
            )
            path = dl.download()
            return os.path.basename(path)
        else:
            ckpt_id, version_id = parse_air(air)
            if not ckpt_id:
                raise ValueError(f"{MSG_PREFIX} AIR inválido: {air}")
            dl = Dreamless_Downloader(
                model_id=ckpt_id,
                model_version=version_id,
                model_types=["Checkpoint"],
                token=civitai_token,
                save_path=download_path,
            )
            if not dl.download():
                raise RuntimeError(f"{MSG_PREFIX} Falha ao baixar checkpoint.")
            return dl.name

    def _download_vae(self, air: str, civitai_token: str, hf_token: str) -> str | None:
        if is_hf(air):
            repo_id, filename, revision = parse_hf(air)
            if not repo_id or not filename:
                return None
            dl = Dreamless_HF_Downloader(
                repo_id=repo_id,
                filename=filename,
                token=hf_token,
                revision=revision,
                save_path=VAE_PATHS[0],
            )
            path = dl.download()
            return os.path.basename(path)
        else:
            vae_id, vae_version_id = parse_air(air)
            if not vae_id:
                return None
            dl = Dreamless_Downloader(
                model_id=vae_id,
                model_version=vae_version_id,
                model_types=["VAE"],
                token=civitai_token,
                save_path=VAE_PATHS[0],
            )
            return dl.name if dl.download() else None

    def _download_lora(self, air: str, civitai_token: str, hf_token: str) -> str | None:
        if is_hf(air):
            repo_id, filename, revision = parse_hf(air)
            if not repo_id or not filename:
                return None
            dl = Dreamless_HF_Downloader(
                repo_id=repo_id,
                filename=filename,
                token=hf_token,
                revision=revision,
                save_path=LORAS[0],
            )
            path = dl.download()
            return os.path.basename(path)
        else:
            lora_id, lora_version_id = parse_air(air)
            if not lora_id:
                return None
            dl = Dreamless_Downloader(
                model_id=lora_id,
                model_version=lora_version_id,
                model_types=["Lora", "LyCORIS"],
                token=civitai_token,
                save_path=LORAS[0],
            )
            return dl.name if dl.download() else None

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
        civitai_api_key="",
        hf_api_key="",
        download_path=None,
        optional_model=None,
        optional_clip=None,
        optional_vae=None,
        optional_lora_stack=None,
        extra_pnginfo=None,
    ):
        civitai_token = self._resolve_token(civitai_api_key, "civitai")
        hf_token = self._resolve_token(hf_api_key, "huggingface")

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
                checkpoint_paths = short_paths_map(CHECKPOINTS)
                resolved_path = (
                    checkpoint_paths[download_path]
                    if download_path and download_path in checkpoint_paths
                    else CHECKPOINTS[0]
                )

                ckpt_name = self._download_ckpt(
                    ckpt_air, civitai_token, hf_token, resolved_path
                )

                if (
                    not is_hf(ckpt_air)
                    and extra_pnginfo
                    and "workflow" in extra_pnginfo
                ):
                    ckpt_id, version_id = parse_air(ckpt_air)
                    extra_pnginfo["workflow"].setdefault("extra", {}).setdefault(
                        "ckpt_airs", []
                    )
                    air = f"{ckpt_id}@{version_id}"
                    if air not in extra_pnginfo["workflow"]["extra"]["ckpt_airs"]:
                        extra_pnginfo["workflow"]["extra"]["ckpt_airs"].append(air)
            else:
                print(f"{MSG_PREFIX}Loading checkpoint: {ckpt_name}")

            model, clip, vae = self.ckpt_loader.load_checkpoint(ckpt_name=ckpt_name)

        if (
            vae_name == "none"
            and vae_air
            and vae_air.strip() not in ("none", "{model_id}@{model_version}")
        ):
            downloaded_vae = self._download_vae(vae_air, civitai_token, hf_token)
            if downloaded_vae:
                vae_name = downloaded_vae

        if optional_vae:
            vae = optional_vae
        elif vae_name not in ("Baked VAE", "none"):
            vae = self.vae_loader.load_vae(vae_name=vae_name)[0]

        if (
            lora_name == "none"
            and lora_air
            and lora_air.strip() not in ("none", "{model_id}@{model_version}")
        ):
            downloaded_lora = self._download_lora(lora_air, civitai_token, hf_token)
            if downloaded_lora:
                lora_name = downloaded_lora

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
                    lora_filename, strength_model, strength_clip = entry
                else:
                    lora_filename = entry.get("name") or entry.get("lora_name")
                    strength_model = entry.get("strength_model", 1.0)
                    strength_clip = entry.get("strength_clip", 1.0)

                if lora_filename and lora_filename != "none":
                    print(f"\33[1m\33[36m[Dreamless] Load LoRA \33[0m {lora_filename}-> model: {strength_model} | clip {strength_clip}")
                    model, clip = self.lora_loader.load_lora(
                        model=model,
                        clip=clip,
                        lora_name=lora_filename,
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
