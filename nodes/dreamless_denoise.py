

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Denoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("DENOISE",)
    FUNCTION = "get_denoise"
    CATEGORY = "Dreamless/Utils"

    def get_denoise(self, denoise):
        return (denoise,)
