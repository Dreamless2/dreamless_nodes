import folder_paths

from nodes import (
    CLIPLoader,
    DualCLIPLoader,
)

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_CLIP_Loader:
    def __init__(self):
        self.clip_loader = None

    @classmethod
    def INPUT_TYPES(cls):
        text_encoders = folder_paths.get_filename_list("text_encoders")
        if not text_encoders:
            text_encoders = ["none"]

        return {
            "required": {
                "clip_name": (text_encoders,),
                "type": (
                    [
                        "stable_diffusion",
                        "stable_cascade",
                        "sd3",
                        "stable_audio",
                        "mochi",
                        "ltxv",
                        "pixart",
                        "cosmos",
                        "lumina2",
                        "wan",
                        "hidream",
                        "chroma",
                        "ace",
                        "omnigen2",
                        "qwen_image",
                        "hunyuan_image",
                        "flux2",
                        "ovis",
                        "longcat_image",
                        "cogvideox",
                        "lens",
                        "pixeldit",
                        "ideogram4",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "Dreamless/Loaders"

    def load_clip(self, clip_name, type):
        if clip_name == "none":
            raise ValueError(f"{MSG_PREFIX}Please select a valid text encoder.")

        if self.clip_loader is None:
            self.clip_loader = CLIPLoader()

        print(f"{MSG_PREFIX}Loading CLIP: {clip_name} ({type})")

        clip = self.clip_loader.load_clip(clip_name=clip_name, type=type)[0]

        return (clip,)


class Dreamless_DualCLIP_Loader:
    def __init__(self):
        self.clip_loader = None

    @classmethod
    def INPUT_TYPES(cls):
        text_encoders = folder_paths.get_filename_list("text_encoders")
        if not text_encoders:
            text_encoders = ["none"]

        return {
            "required": {
                "clip_name1": (text_encoders,),
                "clip_name2": (text_encoders,),
                "type": (
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
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "Dreamless/Loaders"

    def load_clip(self, clip_name1, clip_name2, type):
        if clip_name1 == "none":
            raise ValueError(f"{MSG_PREFIX}Please select the first text encoder.")

        if clip_name2 == "none":
            raise ValueError(f"{MSG_PREFIX}Please select the second text encoder.")

        if self.clip_loader is None:
            self.clip_loader = DualCLIPLoader()

        print(f"{MSG_PREFIX}Loading DualCLIP: {clip_name1} + {clip_name2} ({type})")

        clip = self.clip_loader.load_clip(
            clip_name1=clip_name1,
            clip_name2=clip_name2,
            type=type,
        )[0]

        return (clip,)
