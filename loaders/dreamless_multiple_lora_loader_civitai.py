

import folder_paths
from nodes import LoraLoader

from ..utils.helpers import get_api_key, set_api_key, short_paths_map, parse_air
from ..downloaders.dreamless_downloader import Dreamless_Downloader

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"
MAX_LORAS = 20

LORAS = folder_paths.folder_names_and_paths.get("loras", [["", ""]])[0]
if isinstance(LORAS, str):
    LORAS = [LORAS]


class Dreamless_Multiple_LORA_Loader_CivitAI:
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

        optional = {
            "lora_count": (
                "INT",
                {"default": 1, "min": 1, "max": MAX_LORAS, "step": 1},
            ),
            "api_key": ("STRING", {"default": display_key, "multiline": False}),
            "download_path": (list(lora_paths.keys()),),
        }

        for i in range(MAX_LORAS):
            optional[f"lora_name_{i}"] = (loras,)
            optional[f"strength_model_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
            )
            optional[f"strength_clip_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
            )

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_loras"
    CATEGORY = "Dreamless/Loaders"

    def load_loras(
        self, model, clip, lora_count=1, api_key="", download_path=None, **kwargs
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

        if not self.lora_loader:
            self.lora_loader = LoraLoader()

        current_model = model
        current_clip = clip

        lora_paths = short_paths_map(LORAS)
        resolved_download_path = (
            lora_paths.get(download_path, LORAS[0]) if download_path else LORAS[0]
        )

        for i in range(lora_count):
            lora_name = kwargs.get(f"lora_name_{i}", "none")
            strength_model = float(kwargs.get(f"strength_model_{i}", 1.0))
            strength_clip = float(kwargs.get(f"strength_clip_{i}", 1.0))

            if lora_name == "none":
                continue

            if (
                "@" in lora_name
                or "civitai.com" in lora_name
                or (
                    not lora_name.endswith(".safetensors")
                    and not lora_name.endswith(".ckpt")
                )
            ):
                lora_id, version_id = parse_air(lora_name)
                if lora_id:
                    print(
                        f"{MSG_PREFIX}Slot {i + 1} has an AIR/URL string. Triggering background downloader..."
                    )
                    downloader = Dreamless_Downloader(
                        model_id=lora_id,
                        model_version=version_id,
                        model_types=["Lora", "LyCORIS"],
                        token=token_to_use,
                        save_path=resolved_download_path,
                    )
                    if downloader.download():
                        lora_name = downloader.name
                    else:
                        print(
                            f"{MSG_PREFIX}FAILED to download LoRA for slot {i + 1}. Skipping slot."
                        )
                        continue

            print(f"\33[1m\33[36m[Dreamless] Load LoRA:\33[0m{i + 1}/{lora_count}: {lora_name}: model={strength_model}, clip={strength_clip}")

            current_model, current_clip = self.lora_loader.load_lora(
                model=current_model,
                clip=current_clip,
                lora_name=lora_name,
                strength_model=strength_model,
                strength_clip=strength_clip,
            )

        return (current_model, current_clip)
