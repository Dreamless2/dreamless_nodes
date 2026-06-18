import folder_paths
from nodes import UNETLoader

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Diffusion_Model_Loader:
    def __init__(self):
        self.loader = None

    @classmethod
    def INPUT_TYPES(cls):
        unet_list = folder_paths.get_filename_list("diffusion_models")
        if not unet_list:
            unet_list = ["none"]

        return {
            "required": {
                "unet_name": (unet_list,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_diffusion_model"
    CATEGORY = "Dreamless/Loaders"

    def load_diffusion_model(self, unet_name):
        if unet_name == "none":
            raise ValueError(f"{MSG_PREFIX}Please select a valid diffusion model.")

        if not self.loader:
            self.loader = UNETLoader()

        print(f"{MSG_PREFIX}Loading diffusion model from disk: {unet_name}")
        return self.loader.load_unet(unet_name=unet_name)
