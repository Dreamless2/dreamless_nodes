MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Steps:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("STEPS",)
    FUNCTION = "get_steps"
    CATEGORY = "Dreamless/Utils"

    def get_steps(self, steps):
        return (steps,)
