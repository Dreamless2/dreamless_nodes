import folder_paths

LORAS = folder_paths.folder_names_and_paths["loras"][0]
if isinstance(LORAS, str):
    LORAS = [LORAS]

MSG_PREFIX = "\33[1m\33[34m[Dreamless] \33[0m"
MAX_LORAS = 20


class Dreamless_LORA_Stack:
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["none"] + folder_paths.get_filename_list("loras")

        optional = {
            "lora_stack": ("LORA_STACK",),
            "enable_lora_injection": ("BOOLEAN", {"default": True}),
            "lora_count": (
                "INT",
                {"default": 1, "min": 1, "max": MAX_LORAS, "step": 1},
            ),
        }

        for i in range(MAX_LORAS):
            optional[f"lora_name_{i}"] = (loras,)
            optional[f"strength_model_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
            )
            optional[f"strength_clip_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
            )

        return {
            "required": {},
            "optional": optional,
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("LORA_STACK",)
    FUNCTION = "stack_loras"
    CATEGORY = "Dreamless/Loaders"

    def stack_loras(
        self, lora_count=1, lora_stack=None, enable_lora_injection=True, **kwargs
    ):
        print(f"{MSG_PREFIX}Starting Dreamless LoRA Stack...")

        if not enable_lora_injection:
            print(f"{MSG_PREFIX}LoRA stack bypassed.")
            return ([],)

        result_stack = []

        if lora_stack is not None:
            result_stack.extend(lora_stack)

        for i in range(lora_count):
            lora_name = kwargs.get(f"lora_name_{i}", "none")
            strength_model = float(kwargs.get(f"strength_model_{i}", 1.0))
            strength_clip = float(kwargs.get(f"strength_clip_{i}", 1.0))

            if lora_name == "none":
                continue

            if any(item[0] == lora_name for item in result_stack):
                print(f"{MSG_PREFIX}LoRA already exists.")
                continue

            result_stack.append((lora_name, strength_model, strength_clip))
            print(f"\33[1m\33[36m[Dreamless] Load LoRA \33[0m {lora_filename}-> model: {strength_model} | clip: {strength_clip}")

        return (result_stack,)
