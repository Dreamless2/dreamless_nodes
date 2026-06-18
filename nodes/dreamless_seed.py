import random

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Seed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("SEED",)
    FUNCTION = "get_seed"
    CATEGORY = "Dreamless/Utils"

    def get_seed(self, seed):
        if seed == 0:
            seed = random.randint(1, 0xFFFFFFFFFFFFFFFF)
            print(f"{MSG_PREFIX}Zero seed detected. Generated runtime seed: {seed}")

        return (seed,)
