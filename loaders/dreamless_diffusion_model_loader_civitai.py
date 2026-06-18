import os
import folder_paths
from nodes import UNETLoader
from ..downloaders.dreamless_downloader import Dreamless_Downloader
from ..utils.helpers import short_paths_map, model_path, parse_air, get_api_key

UNETS = folder_paths.folder_names_and_paths["diffusion_models"][0]
if isinstance(UNETS, str):
    UNETS = [UNETS]

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Diffusion_Model_Loader_CivitAI:
    def __init__(self):
        self.loader = None

    @classmethod
    def INPUT_TYPES(cls):
        unet_list = folder_paths.get_filename_list("diffusion_models")
        unet_list.insert(0, "none")

        try:
            unet_paths = short_paths_map(UNETS)
        except:
            unet_paths = {}

        saved_key = get_api_key("civitai")
        display_key = (
            f"{saved_key[:6]}...{saved_key[-4:]}" if len(saved_key) > 10 else saved_key
        )

        return {
            "required": {
                "unet_air": (
                    "STRING",
                    {
                        "default": "{model_id}@{version_id}",
                        "multiline": False,
                    },
                ),
                "unet_name": (unet_list,),
            },
            "optional": {
                "api_key": ("STRING", {"default": display_key, "multiline": False}),
                "download_path": (
                    list(unet_paths.keys()) if unet_paths else ["default"],
                ),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_diffusion_model"
    CATEGORY = "Dreamless/Loaders"

    def load_diffusion_model(
        self,
        unet_air,
        unet_name,
        api_key="",
        download_path=None,
        extra_pnginfo=None,
    ):
        final_path = None
        model_id, version_id = parse_air(unet_air)

        if unet_name == "none":
            if not model_id:
                raise ValueError(
                    f"{MSG_PREFIX}Invalid Diffusion AIR ID or no local model selected."
                )

            resolved_folder = UNETS[0]
            if download_path:
                try:
                    unet_paths = short_paths_map(UNETS)
                    if download_path in unet_paths:
                        resolved_folder = unet_paths[download_path]
                except:
                    pass

            try:
                downloader = Dreamless_Downloader(
                    model_id=model_id,
                    model_version=version_id,
                    model_types=["DiffusionModel", "Checkpoint"],
                    token=api_key,
                    save_path=resolved_folder,
                )
                final_path = downloader.download()
            except Exception as e:
                print(f"{MSG_PREFIX}Multipart download failed: {e}")
                return (None,)

            if final_path and os.path.isfile(final_path):
                unet_name = os.path.basename(final_path)

                if extra_pnginfo and "workflow" in extra_pnginfo:
                    extra_pnginfo["workflow"].setdefault("extra", {}).setdefault(
                        "unet_airs", []
                    )
                    air = f"{downloader.model_id}@{downloader.version}"
                    if air not in extra_pnginfo["workflow"]["extra"]["unet_airs"]:
                        extra_pnginfo["workflow"]["extra"]["unet_airs"].append(air)
        else:
            unet_file_path = model_path(unet_name, UNETS)
            print(f"{MSG_PREFIX}Loading diffusion model from disk: {unet_file_path}")

        if unet_name == "none" and not final_path:
            raise ValueError(f"{MSG_PREFIX}No valid diffusion model path found.")

        if not self.loader:
            self.loader = UNETLoader()

        return self.loader.load_unet(unet_name=unet_name)
