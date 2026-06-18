from nodes import CLIPTextEncode

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_CLIP_Text_Encode:
    def __init__(self):
        self.encoder = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": True, "default": "", "height": 400},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("CONDITIONING",)
    FUNCTION = "encode_text"
    CATEGORY = "Dreamless/Conditioning"

    def encode_text(self, clip, text):
        if clip is None:
            raise ValueError(
                f"{MSG_PREFIX}No valid CLIP model connected to the encoder."
            )

        if not self.encoder:
            self.encoder = CLIPTextEncode()

        print(f"{MSG_PREFIX}Encoding text prompt tokens...")
        return self.encoder.encode(clip=clip, text=text)
