

import folder_paths
from nodes import LoraLoader
from typing import Any, Mapping

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"
MAX_LORAS = 20


class Dreamless_Multiple_LORA_Loader:
    def __init__(self):
        self.lora_loader = None

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["none"] + folder_paths.get_filename_list("loras")

        optional: dict[str, tuple[str, Mapping[str, Any]]] = {
            "lora_count": (
                "INT",
                {"default": 1, "min": 1, "max": MAX_LORAS, "step": 1},
            ),
        }

        for i in range(MAX_LORAS):
            optional[f"lora_name_{i}"] = (
                "COMBO",
                {"values": loras},
            )
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

    def load_loras(self, model, clip, lora_count=1, **kwargs):
        if not self.lora_loader:
            self.lora_loader = LoraLoader()

        current_model = model
        current_clip = clip

        for i in range(lora_count):
            lora_name = kwargs.get(f"lora_name_{i}", "none")
            strength_model = float(kwargs.get(f"strength_model_{i}", 1.0))
            strength_clip = float(kwargs.get(f"strength_clip_{i}", 1.0))

            if lora_name == "none":
                continue

            print(f"\33[1m\33[36m[Dreamless] Load LoRA:\33[0m{i + 1}/{lora_count}: {lora_name}: model={strength_model}, clip={strength_clip}")
            
            current_model, current_clip = self.lora_loader.load_lora(
                current_model,
                current_clip,
                lora_name,
                strength_model,
                strength_clip,
            )

        return (current_model, current_clip)
