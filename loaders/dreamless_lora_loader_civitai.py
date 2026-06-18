

import folder_paths
from nodes import LoraLoader

from ..utils.helpers import get_api_key, set_api_key, short_paths_map, parse_air
from ..downloaders.dreamless_downloader import Dreamless_Downloader

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"

LORAS = folder_paths.folder_names_and_paths.get("loras", [["", ""]])[0]
if isinstance(LORAS, str):
    LORAS = [LORAS]


class Dreamless_LORA_Loader_CivitAI:
    def __init__(self):
        self.lora_loader = None

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["none"] + folder_paths.get_filename_list("loras")
        lora_paths = short_paths_map(LORAS)

        saved_key = get_api_key("civitai")
        display_key = (
            f"{saved_key[:6]}...{saved_key[-4:]}" if len(saved_key) > 10 else saved_key
        )

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_air": (
                    "STRING",
                    {"default": "{model_id}@{model_version}", "multiline": False},
                ),
                "lora_name": (loras,),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            },
            "optional": {
                "api_key": ("STRING", {"default": display_key, "multiline": False}),
                "download_path": (list(lora_paths.keys()),),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "Dreamless/Loaders"

    def load_lora(
        self,
        model,
        clip,
        lora_air,
        lora_name,
        strength_model,
        strength_clip,
        api_key="",
        download_path=None,
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

        if lora_name == "none":
            lora_id, version_id = parse_air(lora_air)
            if not lora_id:
                raise ValueError(
                    f"{MSG_PREFIX}Please provide a valid LoRA AIR or select a local file."
                )

            lora_paths = short_paths_map(LORAS)
            resolved_download_path = (
                lora_paths.get(download_path, LORAS[0]) if download_path else LORAS[0]
            )

            downloader = Dreamless_Downloader(
                model_id=lora_id,
                model_version=version_id,
                model_types=["Lora", "LyCORIS"],
                token=token_to_use,
                save_path=resolved_download_path,
            )
            if not downloader.download():
                raise RuntimeError(f"{MSG_PREFIX}Failed to download LoRA.")
            lora_name = downloader.name
        else:
            print(f"{MSG_PREFIX}Loading LoRA from disk: {lora_name}")

        if not self.lora_loader:
            self.lora_loader = LoraLoader()

        return self.lora_loader.load_lora(
            model=model,
            clip=clip,
            lora_name=lora_name,
            strength_model=strength_model,
            strength_clip=strength_clip,
        )
