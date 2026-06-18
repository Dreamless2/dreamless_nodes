import folder_paths
import os

from ..downloaders.dreamless_downloader import Dreamless_Downloader
from ..utils.helpers import (
    get_api_key,
    set_api_key,
    parse_air,
)

LORAS = folder_paths.folder_names_and_paths["loras"][0]
if isinstance(LORAS, str):
    LORAS = [LORAS]

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"
MAX_LORAS = 20


class Dreamless_LORA_Stack_CivitAI:
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["none"] + folder_paths.get_filename_list("loras")

        saved_key = get_api_key("civitai")
        display_key = (
            f"{saved_key[:6]}...{saved_key[-4:]}" if len(saved_key) > 10 else saved_key
        )

        optional = {
            "lora_stack": ("LORA_STACK",),
            "enable_lora_injection": ("BOOLEAN", {"default": True}),
            "lora_count": (
                "INT",
                {"default": 1, "min": 1, "max": MAX_LORAS, "step": 1},
            ),
            "api_key": (
                "STRING",
                {"default": display_key, "multiline": False},
            ),
        }

        for i in range(MAX_LORAS):
            optional[f"lora_air_{i}"] = (
                "STRING",
                {"default": "{model_id}@{version_id}", "multiline": False},
            )
            optional[f"lora_name_{i}"] = (loras,)
            optional[f"strength_model_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
            )
            optional[f"strength_clip_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
            )

        return {
            "required": {},
            "optional": optional,
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("LORA_STACK",)
    FUNCTION = "stack_loras"
    CATEGORY = "Dreamless/Loaders"

    def stack_loras(
        self,
        lora_count=1,
        lora_stack=None,
        enable_lora_injection=True,
        api_key="",
        **kwargs,
    ):
        if not enable_lora_injection:
            return ([],)

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

        result_stack = []

        if lora_stack is not None:
            result_stack.extend(lora_stack)

        for i in range(lora_count):
            lora_air = kwargs.get(f"lora_air_{i}", "none")
            lora_name = kwargs.get(f"lora_name_{i}", "none")
            strength_model = float(kwargs.get(f"strength_model_{i}", 1.0))
            strength_clip = float(kwargs.get(f"strength_clip_{i}", 1.0))

            if lora_name == "none" and lora_air and lora_air.strip().lower() != "none":
                lora_id, lora_version = parse_air(lora_air)

                if lora_id:
                    downloader = Dreamless_Downloader(
                        model_id=lora_id,
                        model_version=lora_version,
                        model_types=["LORA", "Lora", "LyCORIS"],
                        token=token_to_use,
                        save_path=LORAS[0],
                    )

                    try:
                        filepath = downloader.download()
                        if filepath:
                            lora_name = os.path.basename(filepath)
                            print(f"\33[1m\33[33m[Dreamless] Using downloaded LoRA:\33[0m {lora_name}")
                    except Exception as e:
                        print(f"{MSG_PREFIX}Failed to download LoRA: {e}")
                        continue

            if lora_name == "none":
                continue

            if any(item[0] == lora_name for item in result_stack):
                print(f"{MSG_PREFIX}LoRA already exists.")
                continue

            result_stack.append((lora_name, strength_model, strength_clip))
            print(f"\33[1m\33[36m[Dreamless] Load LoRA:\33[0m {lora_name}: model={strength_model}, clip={strength_clip}")

        return (result_stack,)
