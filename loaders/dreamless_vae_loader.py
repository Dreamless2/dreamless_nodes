import folder_paths
from nodes import VAELoader

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_VAE_Loader:
    def __init__(self):
        self.loader = None

    @classmethod
    def INPUT_TYPES(cls):
        vae_list = folder_paths.get_filename_list("vae")
        if not vae_list:
            vae_list = ["none"]

        return {
            "required": {
                "vae_name": (vae_list,),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load_vae_model"
    CATEGORY = "Dreamless/Loaders"

    def load_vae_model(self, vae_name):
        if vae_name == "none":
            raise ValueError(f"{MSG_PREFIX}Please select a valid VAE model.")

        if not self.loader:
            self.loader = VAELoader()

        print(f"{MSG_PREFIX}Loading VAE from disk: {vae_name}")
        return self.loader.load_vae(vae_name=vae_name)
