

import folder_paths
from nodes import LoraLoader

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_LORA_Loader:
    def __init__(self):
        self.lora_loader = None

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        loras.insert(0, "none")

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (loras,),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "Dreamless/Loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if lora_name == "none":
            return model, clip

        if not self.lora_loader:
            self.lora_loader = LoraLoader()

        print(
            f"\33[1m\33[36m[Dreamless] Load LoRA:\33[0m {lora_name}: model={strength_model}, clip={strength_clip}"
        )
        return self.lora_loader.load_lora(
            model, clip, lora_name, strength_model, strength_clip
        )
