

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_CFG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cfg": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("CFG",)
    FUNCTION = "get_cfg"
    CATEGORY = "Dreamless/Utils"

    def get_cfg(self, cfg):
        return (cfg,)
