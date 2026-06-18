import comfy.samplers

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Sampler_Selector:
    @classmethod
    def INPUT_TYPES(cls):
        # Puxa dinamicamente todos os samplers registados no sistema
        samplers_list = comfy.samplers.KSampler.SAMPLERS
        return {
            "required": {
                "sampler_name": (samplers_list, {"default": "lcm"}),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("SAMPLER_NAME",)
    FUNCTION = "get_sampler_name"
    CATEGORY = "Dreamless/Utils"

    def get_sampler_name(self, sampler_name):
        return (sampler_name,)
