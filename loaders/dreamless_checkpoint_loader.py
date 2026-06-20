

import folder_paths 
from nodes import CheckpointLoaderSimple

CHECKPOINTS = folder_paths.folder_names_and_paths["checkpoints"][0]
if isinstance(CHECKPOINTS, str):
    CHECKPOINTS = [CHECKPOINTS]

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Checkpoint_Loader:
    def __init__(self):
        self.ckpt_loader = None

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        if not checkpoints:
            checkpoints = ["none"]

        return {
            "required": {
                "ckpt_name": (checkpoints,),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "Dreamless/Loaders"

    def load_checkpoint(self, ckpt_name):
        if ckpt_name == "none":
            raise ValueError(f"{MSG_PREFIX}Please select a valid checkpoint.")

        if not self.ckpt_loader:
            self.ckpt_loader = CheckpointLoaderSimple()

        print(f"\33[1m\33[36m[Dreamless] Load checkpoint \33[0m -> {ckpt_name}")
        return self.ckpt_loader.load_checkpoint(ckpt_name=ckpt_name)
