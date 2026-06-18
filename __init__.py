import traceback

MSG_PREFIX = "\033[1m\033[35m[Dreamless]\033[0m"
VERSION = "1.0.0"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
PURPLE = "\033[1m\033[35m"
WHITE = "\033[1m\033[37m"
YELLOW = "\033[1m\033[33m"
GREEN = "\033[1m\033[32m"
RESET = "\033[0m"

try:
    # loaders
    from .loaders.dreamless_loader import Dreamless_Loader
    from .loaders.dreamless_loader_civitai import Dreamless_Loader_CivitAI
    from .loaders.dreamless_lora_loader import Dreamless_LORA_Loader
    from .loaders.dreamless_multiple_lora_loader import Dreamless_Multiple_LORA_Loader
    from .loaders.dreamless_diffusion_model_loader import (
        Dreamless_Diffusion_Model_Loader,
    )
    from .loaders.dreamless_diffusion_model_loader_civitai import (
        Dreamless_Diffusion_Model_Loader_CivitAI,
    )
    from .loaders.dreamless_diffusion_model_dualclip_loader import (
        Dreamless_Diffusion_Model_DualCLIP_Loader,
    )
    from .loaders.dreamless_diffusion_model_dualclip_loader_civitai import (
        Dreamless_Diffusion_Model_DualCLIP_Loader_CivitAI,
    )
    from .loaders.dreamless_diffusion_model_clip_loader_civitai import (
        Dreamless_Diffusion_Model_CLIP_Loader_CivitAI,
    )
    from .loaders.dreamless_checkpoint_loader import Dreamless_Checkpoint_Loader
    from .loaders.dreamless_checkpoint_loader_civitai import (
        Dreamless_Checkpoint_Loader_CivitAI,
    )
    from .loaders.dreamless_lora_loader_civitai import Dreamless_LORA_Loader_CivitAI
    from .loaders.dreamless_multiple_lora_loader_civitai import (
        Dreamless_Multiple_LORA_Loader_CivitAI,
    )
    from .loaders.dreamless_loader_singleton import Dreamless_Loader_Singleton

    # stacks
    from .stacks.dreamless_lora_stack import Dreamless_LORA_Stack
    from .stacks.dreamless_lora_stack_civitai import Dreamless_LORA_Stack_CivitAI

    # nodes
    from .nodes.dreamless_seed import Dreamless_Seed
    from .nodes.dreamless_sampler_selector import Dreamless_Sampler_Selector
    from .nodes.dreamless_scheduler_selector import Dreamless_Scheduler_Selector
    from .nodes.dreamless_steps import Dreamless_Steps
    from .nodes.dreamless_cfg import Dreamless_CFG
    from .nodes.dreamless_denoise import Dreamless_Denoise
    from .nodes.dreamless_vae_nodes import (
        Dreamless_VAE_Decode,
        Dreamless_VAE_Encode,
        Dreamless_VAE_Tiled,
    )
    from .nodes.dreamless_clip_nodes import (
        Dreamless_CLIP_Loader,
        Dreamless_DualCLIP_Loader,
    )
    from .nodes.dreamless_upscale_image import Dreamless_Upscale_Image
    from .nodes.dreamless_upscale_image_by import Dreamless_Upscale_Image_By
    from .nodes.dreamless_upscale_image_by_model import Dreamless_Upscale_Image_By_Model
    from .nodes.dreamless_upscale_image_model import Dreamless_Upscale_Image_Model

    # latent
    from .latent.dreamless_smart_latent import Dreamless_Smart_Latent

    # extensions
    from .extensions.dreamless_context import Dreamless_Context

    # images
    from .images.dreamless_image_adjustments import Dreamless_Image_Adjustments
    from .images.dreamless_image_loader import Dreamless_Image_Loader
    from .images.dreamless_save_image_advanced import Dreamless_Save_Image_Advanced
    from .images.dreamless_save_image_preview import Dreamless_Save_Image_Preview
    from .images.dreamless_preview_image import Dreamless_Preview_Image
    from .images.dreamless_save_image import Dreamless_Save_Image

    # conditioning
    from .conditioning.dreamless_clip_text_encode import Dreamless_CLIP_Text_Encode

    # samplers
    from .samplers.dreamless_ksampler_full import Dreamless_KSampler_Full
    from .samplers.dreamless_ksampler_simple import Dreamless_KSampler_Simple
    from .samplers.dreamless_ksampler_hires import Dreamless_KSampler_Hires
    from .samplers.dreamless_ksampler_config_simple import (
        Dreamless_KSampler_Config_Simple,
    )
    from .samplers.dreamless_ksamplers import (
        Dreamless_KSampler_Config_Advanced,
        Dreamless_KSampler_Standard,
        Dreamless_KSampler_Advanced,
    )

    NODE_CLASS_MAPPINGS = {
        # loaders
        "Dreamless_Loader": Dreamless_Loader,
        "Dreamless_Loader_CivitAI": Dreamless_Loader_CivitAI,
        "Dreamless_LORA_Loader": Dreamless_LORA_Loader,
        "Dreamless_Multiple_LORA_Loader": Dreamless_Multiple_LORA_Loader,
        "Dreamless_LORA_Stack": Dreamless_LORA_Stack,
        "Dreamless_KSampler_Full": Dreamless_KSampler_Full,
        "Dreamless_KSampler_Simple": Dreamless_KSampler_Simple,
        "Dreamless_KSampler_Hires": Dreamless_KSampler_Hires,
        "Dreamless_KSampler_Config_Simple": Dreamless_KSampler_Config_Simple,
        "Dreamless_Seed": Dreamless_Seed,
        "Dreamless_Sampler_Selector": Dreamless_Sampler_Selector,
        "Dreamless_Scheduler_Selector": Dreamless_Scheduler_Selector,
        "Dreamless_Steps": Dreamless_Steps,
        "Dreamless_CFG": Dreamless_CFG,
        "Dreamless_Denoise": Dreamless_Denoise,
        "Dreamless_Smart_Latent": Dreamless_Smart_Latent,
        "Dreamless_Image_Adjustments": Dreamless_Image_Adjustments,
        "Dreamless_Image_Loader": Dreamless_Image_Loader,
        "Dreamless_Save_Image_Advanced": Dreamless_Save_Image_Advanced,
        "Dreamless_Save_Image_Preview": Dreamless_Save_Image_Preview,
        "Dreamless_VAE_Decode": Dreamless_VAE_Decode,
        "Dreamless_VAE_Encode": Dreamless_VAE_Encode,
        "Dreamless_VAE_Tiled": Dreamless_VAE_Tiled,
        "Dreamless_Context": Dreamless_Context,
        "Dreamless_LORA_Stack_CivitAI": Dreamless_LORA_Stack_CivitAI,
        "Dreamless_Diffusion_Model_Loader": Dreamless_Diffusion_Model_Loader,
        "Dreamless_Diffusion_Model_Loader_CivitAI": Dreamless_Diffusion_Model_Loader_CivitAI,
        "Dreamless_Diffusion_Model_DualCLIP_Loader": Dreamless_Diffusion_Model_DualCLIP_Loader,
        "Dreamless_Diffusion_Model_DualCLIP_Loader_CivitAI": Dreamless_Diffusion_Model_DualCLIP_Loader_CivitAI,
        "Dreamless_CLIP_Loader": Dreamless_CLIP_Loader,
        "Dreamless_DualCLIP_Loader": Dreamless_DualCLIP_Loader,
        "Dreamless_Diffusion_Model_CLIP_Loader_CivitAI": Dreamless_Diffusion_Model_CLIP_Loader_CivitAI,
        "Dreamless_CLIP_Text_Encode": Dreamless_CLIP_Text_Encode,
        "Dreamless_Checkpoint_Loader": Dreamless_Checkpoint_Loader,
        "Dreamless_Checkpoint_Loader_CivitAI": Dreamless_Checkpoint_Loader_CivitAI,
        "Dreamless_Preview_Image": Dreamless_Preview_Image,
        "Dreamless_LORA_Loader_CivitAI": Dreamless_LORA_Loader_CivitAI,
        "Dreamless_Multiple_LORA_Loader_CivitAI": Dreamless_Multiple_LORA_Loader_CivitAI,
        "Dreamless_Loader_Singleton": Dreamless_Loader_Singleton,
        "Dreamless_Upscale_Image": Dreamless_Upscale_Image,
        "Dreamless_Upscale_Image_By": Dreamless_Upscale_Image_By,
        "Dreamless_Upscale_Image_By_Model": Dreamless_Upscale_Image_By_Model,
        "Dreamless_Upscale_Image_Model": Dreamless_Upscale_Image_Model,
        "Dreamless_Save_Image": Dreamless_Save_Image,
        "Dreamless_KSampler_Config_Advanced": Dreamless_KSampler_Config_Advanced,
        "Dreamless_KSampler_Standard": Dreamless_KSampler_Standard,
        "Dreamless_KSampler_Advanced": Dreamless_KSampler_Advanced,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "Dreamless_Loader": "DreamLoader (Full)",
        "Dreamless_Loader_CivitAI": "DreamLoader (CivitAI Downloader)",
        "Dreamless_LORA_Loader": "DreamLoraLoader (Model and CLIP)",
        "Dreamless_LORA_Stack": "DreamLoraStack",
        "Dreamless_Multiple_LORA_Loader": "DreamLoraLoader (Multiple Model and CLIP)",
        "Dreamless_KSampler_Full": "DreamKSampler (Full)",
        "Dreamless_KSampler_Simple": "DreamKSampler (Simple)",
        "Dreamless_KSampler_Hires": "DreamKSampler (Hires Fix)",
        "Dreamless_KSampler_Config_Simple": "DreamKSamplerConfig (Simple)",
        "Dreamless_Seed": "DreamSeed",
        "Dreamless_Sampler_Selector": "DreamSamplerSelector",
        "Dreamless_Scheduler_Selector": "DreamSchedulerSelector",
        "Dreamless_Steps": "DreamSteps",
        "Dreamless_CFG": "DreamCFG",
        "Dreamless_Denoise": "DreamDenoise",
        "Dreamless_Smart_Latent": "DreamSmartLatent",
        "Dreamless_Image_Adjustments": "DreamImageAdjustments",
        "Dreamless_Image_Loader": "DreamLoadImage",
        "Dreamless_Save_Image_Advanced": "DreamSaveImage (Advanced)",
        "Dreamless_Save_Image_Preview": "DreamSaveImage (w/ Preview)",
        "Dreamless_VAE_Decode": "DreamVAEDecode",
        "Dreamless_VAE_Encode": "DreamVAEEncode",
        "Dreamless_VAE_Tiled": "DreamSmartVAETiled",
        "Dreamless_Context": "DreamContext",
        "Dreamless_LORA_Stack_CivitAI": "DreamLoraStack (CivitAI Downloader)",
        "Dreamless_Diffusion_Model_Loader": "DreamDiffusionModelLoader",
        "Dreamless_Diffusion_Model_Loader_CivitAI": "DreamDiffusionModelLoader (CivitAI Downloader)",
        "Dreamless_Diffusion_Model_DualCLIP_Loader": "DreamDiffusionModelLoader (Dual CLIP)",
        "Dreamless_Diffusion_Model_DualCLIP_Loader_CivitAI": "DreamDiffusionModelLoader (Dual CLIP - CivitAI Downloader)",
        "Dreamless_CLIP_Loader": "DreamCLIPLoader",
        "Dreamless_DualCLIP_Loader": "DreamDualCLIPLoader",
        "Dreamless_Diffusion_Model_CLIP_Loader_CivitAI": "DreamDiffusionModelLoader (CLIP - CivitAI Downloader)",
        "Dreamless_CLIP_Text_Encode": "DreamCLIPTextEncode (Prompt)",
        "Dreamless_Checkpoint_Loader": "DreamCheckpointLoader",
        "Dreamless_Checkpoint_Loader_CivitAI": "DreamCheckpointLoader (CivitAIDownloader)",
        "Dreamless_Preview_Image": "DreamPreviewImage",
        "Dreamless_LORA_Loader_CivitAI": "DreamLORALoader (CivitAIDownloader)",
        "Dreamless_Multiple_LORA_Loader_CivitAI": "DreamMultipleLORALoader (CivitAI Downloader)",
        "Dreamless_Loader_Singleton": "DreamLoader (CivitAI / HF Downloader)",
        "Dreamless_Upscale_Image": "DreamUpscaleImage",
        "Dreamless_Upscale_Image_By": "DreamUpscaleImageBy",
        "Dreamless_Upscale_Image_By_Model": "DreamTiledUpscaleImageByModel",
        "Dreamless_Upscale_Image_Model": "DreamTiledUpscaleImageModel",
        "Dreamless_Save_Image": "DreamSaveImage",
        "Dreamless_KSampler_Config_Advanced": "DreamKSamplerConfig (Advanced)",
        "Dreamless_KSampler_Standard": "DreamKSampler (Standard)",
        "Dreamless_KSampler_Advanced": "DreamKSampler (Advanced)",
    }

    node_count = len(NODE_CLASS_MAPPINGS)

    print(
        f"{MSG_PREFIX} {YELLOW}v{VERSION}{RESET} | {GREEN}Initializing{PURPLE} {node_count}{RESET}{GREEN} nodes..."
    )

    print(
        f"{GREEN}[INFO]{RESET} Available nodes: "
        + ", ".join(f"'{name}'" for name in NODE_CLASS_MAPPINGS)
    )

    print(
        f"{MSG_PREFIX} "
        f"{YELLOW}Loaded{RESET} "
        f"{GREEN}{node_count}{RESET} "
        f"{YELLOW}nodes successfully!{RESET} 🎉"
    )

except Exception:
    print("\n\033[1m\033[31m[Dreamless - Importation Error]\033[0m")
    traceback.print_exc()
    print("\n")

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
