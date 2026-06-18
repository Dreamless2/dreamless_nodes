import folder_paths

from nodes import (
    UNETLoader,
    DualCLIPLoader,
    VAELoader,
)

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Diffusion_Model_DualCLIP_Loader:
    def __init__(self):
        self.unet_loader = None
        self.clip_loader = None
        self.vae_loader = None

    @classmethod
    def INPUT_TYPES(cls):
        unet_list = folder_paths.get_filename_list("diffusion_models")
        if not unet_list:
            unet_list = ["none"]

        text_encoders = folder_paths.get_filename_list("text_encoders")
        if not text_encoders:
            text_encoders = ["none"]

        vaes = folder_paths.get_filename_list("vae")
        if not vaes:
            vaes = ["none"]

        return {
            "required": {
                "unet_name": (unet_list,),
                "clip_name1": (text_encoders,),
                "clip_name2": (text_encoders,),
                "clip_type": (
                    [
                        "sdxl",
                        "sd3",
                        "flux",
                        "hunyuan_video",
                        "hidream",
                        "hunyuan_image",
                        "hunyuan_video_15",
                        "kandinsky5",
                        "kandinsky5_image",
                        "ltxv",
                        "newbie",
                        "ace",
                    ],
                ),
                "vae_name": (vaes,),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
        "VAE",
    )

    RETURN_NAMES = (
        "MODEL",
        "CLIP",
        "VAE",
    )

    FUNCTION = "load_model"

    CATEGORY = "Dreamless/Loaders"

    def load_model(
        self,
        unet_name,
        clip_name1,
        clip_name2,
        clip_type,
        vae_name,
    ):
        if unet_name == "none":
            raise ValueError(f"{MSG_PREFIX}Please select a valid diffusion model.")

        if self.unet_loader is None:
            self.unet_loader = UNETLoader()

        print(f"{MSG_PREFIX}Loading diffusion model: {unet_name}")

        model = self.unet_loader.load_unet(unet_name=unet_name)[0]

        if clip_name1 == "none":
            raise ValueError(f"{MSG_PREFIX}Please select the first text encoder.")

        if clip_name2 == "none":
            raise ValueError(f"{MSG_PREFIX}Please select the second text encoder.")

        if self.clip_loader is None:
            self.clip_loader = DualCLIPLoader()

        print(
            f"{MSG_PREFIX}Loading dual clip: "
            f"{clip_name1} + {clip_name2} "
            f"({clip_type})"
        )

        clip = self.clip_loader.load_clip(
            clip_name1=clip_name1,
            clip_name2=clip_name2,
            type=clip_type,
        )[0]

        if vae_name == "none":
            raise ValueError(f"{MSG_PREFIX}Please select a valid VAE.")

        if self.vae_loader is None:
            self.vae_loader = VAELoader()

        print(f"{MSG_PREFIX}Loading VAE: {vae_name}")

        vae = self.vae_loader.load_vae(vae_name=vae_name)[0]

        return (
            model,
            clip,
            vae,
        )
