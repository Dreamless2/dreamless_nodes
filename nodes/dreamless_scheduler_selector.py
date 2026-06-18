import comfy.samplers

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"


class Dreamless_Scheduler_Selector:
    @classmethod
    def INPUT_TYPES(cls):
        schedulers_list = comfy.samplers.KSampler.SCHEDULERS
        return {
            "required": {
                "scheduler": (schedulers_list, {"default": "normal"}),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("SCHEDULER",)
    FUNCTION = "get_scheduler_name"
    CATEGORY = "Dreamless/Utils"

    def get_scheduler_name(self, scheduler):
        return (scheduler,)
