

import folder_paths
from nodes import CheckpointLoaderSimple

from ..utils.helpers import get_api_key, set_api_key, short_paths_map, parse_air
from ..downloaders.dreamless_downloader import Dreamless_Downloader

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"

CHECKPOINTS = folder_paths.folder_names_and_paths["checkpoints"][0]
if isinstance(CHECKPOINTS, str):
    CHECKPOINTS = [CHECKPOINTS]


class Dreamless_Checkpoint_Loader_CivitAI:
    def __init__(self):
        self.ckpt_loader = None

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = ["none"] + folder_paths.get_filename_list("checkpoints")
        checkpoint_paths = short_paths_map(CHECKPOINTS)

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
            },
            "optional": {
                "api_key": ("STRING", {"default": display_key, "multiline": False}),
                "download_path": (list(checkpoint_paths.keys()),),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "Dreamless/Loaders"

    def load_checkpoint(
        self,
        ckpt_air,
        ckpt_name,
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

        if ckpt_name == "none":
            ckpt_id, version_id = parse_air(ckpt_air)
            if not ckpt_id:
                raise ValueError(
                    f"{MSG_PREFIX}Please provide a valid checkpoint AIR or select a local file."
                )

            checkpoint_paths = short_paths_map(CHECKPOINTS)
            resolved_download_path = (
                checkpoint_paths.get(download_path, CHECKPOINTS[0])
                if download_path
                else CHECKPOINTS[0]
            )

            downloader = Dreamless_Downloader(
                model_id=ckpt_id,
                model_version=version_id,
                model_types=["Checkpoint"],
                token=token_to_use,
                save_path=resolved_download_path,
            )
            if not downloader.download():
                raise RuntimeError(f"{MSG_PREFIX}Failed to download checkpoint.")
            ckpt_name = downloader.name
        else:
            print(f"\33[1m\33[36m[Dreamless] Load checkpoint \33[0m -> {ckpt_name}")

        if not self.ckpt_loader:
            self.ckpt_loader = CheckpointLoaderSimple()

        return self.ckpt_loader.load_checkpoint(ckpt_name=ckpt_name)