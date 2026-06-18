import os
import folder_paths
from nodes import VAELoader
from ..downloaders.dreamless_downloader import Dreamless_Downloader
from ..utils.helpers import short_paths_map, model_path, parse_air, get_api_key

VAE_PATHS = folder_paths.folder_names_and_paths.get("vae", [["", ""]])[0]
if isinstance(VAE_PATHS, str):
    VAE_PATHS = [VAE_PATHS]

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_VAE_Loader_CivitAI:
    def __init__(self):
        self.loader = None

    @classmethod
    def INPUT_TYPES(cls):
        vae_list = folder_paths.get_filename_list("vae")
        vae_list.insert(0, "none")

        try:
            vae_paths = short_paths_map(VAE_PATHS)
        except:
            vae_paths = {}

        saved_key = get_api_key("civitai")
        display_key = (
            f"{saved_key[:6]}...{saved_key[-4:]}" if len(saved_key) > 10 else saved_key
        )

        return {
            "required": {
                "vae_air": (
                    "STRING",
                    {
                        "default": "{model_id}@{version_id}",
                        "multiline": False,
                    },
                ),
                "vae_name": (vae_list,),
            },
            "optional": {
                "api_key": (
                    "STRING",
                    {"default": display_key, "multiline": False},
                ),
                "download_path": (
                    list(vae_paths.keys()) if vae_paths else ["default"],
                ),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load_vae_model"
    CATEGORY = "Dreamless/Loaders"

    def load_vae_model(
        self,
        vae_air,
        vae_name,
        api_key="",
        download_path=None,
        extra_pnginfo=None,
    ):
        final_path = None
        model_id, version_id = parse_air(vae_air)

        if vae_name == "none":
            if not model_id:
                raise ValueError(
                    f"{MSG_PREFIX}Invalid VAE AIR ID or no local VAE selected."
                )

            resolved_folder = VAE_PATHS[0]
            if download_path:
                try:
                    vae_paths = short_paths_map(VAE_PATHS)
                    if download_path in vae_paths:
                        resolved_folder = vae_paths[download_path]
                except:
                    pass

            print(f"{MSG_PREFIX}Downloading selected VAE...")
            try:
                downloader = Dreamless_Downloader(
                    model_id=model_id,
                    model_version=version_id,
                    model_types=["VAE"],
                    token=api_key,
                    save_path=resolved_folder,
                )
                final_path = downloader.download()
            except Exception as e:
                print(f"{MSG_PREFIX}Multipart VAE download failed: {e}")
                return (None,)

            if final_path and os.path.isfile(final_path):
                vae_name = os.path.basename(final_path)

                if extra_pnginfo and "workflow" in extra_pnginfo:
                    extra_pnginfo["workflow"].setdefault("extra", {}).setdefault(
                        "vae_airs", []
                    )
                    air = f"{downloader.model_id}@{downloader.version}"
                    if air not in extra_pnginfo["workflow"]["extra"]["vae_airs"]:
                        extra_pnginfo["workflow"]["extra"]["vae_airs"].append(air)
        else:
            vae_file_path = model_path(vae_name, VAE_PATHS)
            print(f"{MSG_PREFIX}Loading VAE from disk: {vae_file_path}")

        if vae_name == "none" and not final_path:
            raise ValueError(f"{MSG_PREFIX}No valid VAE path found.")

        if not self.loader:
            self.loader = VAELoader()

        return self.loader.load_vae(vae_name=vae_name)
