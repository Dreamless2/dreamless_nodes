import os
import re
import json
import time
import folder_paths
import numpy as np

from PIL import Image, PngImagePlugin

INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\n\r\t]'


class Dreamless_Save_Image_Advanced:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "Dreamless_%date:yyyy-MM-dd-hhmmss%",
                        "multiline": False,
                    },
                ),
                "directory_name": (
                    "STRING",
                    {
                        "default": folder_paths.get_output_directory(),
                        "multiline": False,
                    },
                ),
                "subdirectory_name": (
                    "STRING",
                    {
                        "default": "%date:yyyy-MM-dd%",
                        "multiline": False,
                    },
                ),
                "output_format": (
                    ["png", "jpg", "webp"],
                    {"default": "png"},
                ),
                "quality": (
                    "INT",
                    {"default": 80, "min": 1, "max": 100, "step": 1},
                ),
                "metadata_scope": (
                    ["default", "workflow_only", "none"],
                    {"default": "default"},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Dreamless/IO"

    # =========================================================
    # MAIN
    # =========================================================

    def save_images(
        self,
        images,
        filename_prefix,
        directory_name,
        subdirectory_name,
        output_format,
        quality,
        metadata_scope,
        prompt=None,
        extra_pnginfo=None,
    ):
        now = time.localtime()

        width = images[0].shape[1]
        height = images[0].shape[0]

        metadata_map = {
            "width": width,
            "height": height,
            "date": now,
        }

        # =====================================================
        # PATHS
        # =====================================================

        parsed_subdir = self.parse_masks(subdirectory_name, metadata_map)
        parsed_filename = self.parse_masks(filename_prefix, metadata_map)
        parsed_subdir = self.sanitize_filename(parsed_subdir)
        parsed_filename = self.sanitize_filename(parsed_filename)

        full_output_dir = os.path.join(directory_name, parsed_subdir)
        os.makedirs(full_output_dir, exist_ok=True)

        results = []

        # =====================================================
        # SAVE
        # =====================================================

        for batch_number, image in enumerate(images):
            image_np = image.cpu().numpy()
            image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(image_np)

            final_filename = f"{parsed_filename}_{batch_number:03}.{output_format}"
            final_filename = self.limit_filename(final_filename)
            save_path = os.path.join(full_output_dir, final_filename)

            # -------------------------------------------------
            # PNG
            # -------------------------------------------------

            if output_format == "png":
                png_metadata = None

                if metadata_scope != "none":
                    png_metadata = PngImagePlugin.PngInfo()
                    self.add_png_metadata(
                        png_metadata,
                        metadata_scope,
                        prompt,
                        extra_pnginfo,
                    )

                compress_level = max(0, min(9, int((100 - quality) / 10)))
                img.save(save_path, pnginfo=png_metadata, compress_level=compress_level)

            # -------------------------------------------------
            # JPG / WEBP
            # -------------------------------------------------

            else:
                img = img.convert("RGB")

                metadata_json = self.build_json_metadata(
                    metadata_scope,
                    prompt,
                    extra_pnginfo,
                )

                save_kwargs = {"quality": quality, "optimize": True}

                if metadata_json:
                    save_kwargs["comment"] = metadata_json.encode("utf-8")

                if output_format == "jpg":
                    img.save(save_path, format="JPEG", **save_kwargs)
                elif output_format == "webp":
                    img.save(save_path, format="WEBP", method=6, **save_kwargs)

            print(f"[Dreamless] Saved: {save_path}")

            try:
                rel_subfolder = os.path.relpath(full_output_dir, self.output_dir)
                if rel_subfolder.startswith(".."):
                    rel_subfolder = parsed_subdir
            except ValueError:
                # Drives diferentes no Windows (ex: D:\ vs C:\)
                rel_subfolder = parsed_subdir

            results.append(
                {
                    "filename": final_filename,
                    "subfolder": rel_subfolder,
                    "type": self.type,
                }
            )

        return {"ui": {"images": results}}

    # =========================================================
    # METADATA
    # =========================================================

    def add_png_metadata(
        self,
        png_metadata,
        metadata_scope,
        prompt,
        extra_pnginfo,
    ):
        if metadata_scope in ("default",):
            if prompt is not None:
                png_metadata.add_text("prompt", json.dumps(prompt))

        if metadata_scope in ("default", "workflow_only"):
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    png_metadata.add_text(key, json.dumps(value))

    def build_json_metadata(
        self,
        metadata_scope,
        prompt,
        extra_pnginfo,
    ):
        if metadata_scope == "none":
            return None

        metadata = {}

        if metadata_scope in ("default",):
            metadata["prompt"] = prompt

        if metadata_scope in ("default", "workflow_only"):
            metadata["workflow"] = extra_pnginfo

        return json.dumps(metadata, ensure_ascii=False)

    # =========================================================
    # MASKS
    # =========================================================

    def parse_masks(self, text, metadata):
        now = metadata["date"]

        date_patterns = {
            "yyyy": time.strftime("%Y", now),
            "MM": time.strftime("%m", now),
            "dd": time.strftime("%d", now),
            "hh": time.strftime("%H", now),
            "mm": time.strftime("%M", now),
            "ss": time.strftime("%S", now),
        }

        for fmt in re.findall(r"%date:([^%]+)%", text):
            parsed = fmt
            for key, value in date_patterns.items():
                parsed = parsed.replace(key, value)
            text = text.replace(f"%date:{fmt}%", parsed)

        text = text.replace("%date%", time.strftime("%Y%m%d%H%M%S", now))

        tokens = {
            "%width%": str(metadata["width"]),
            "%height%": str(metadata["height"]),
        }

        for token, value in tokens.items():
            text = text.replace(token, value)

        return text

    # =========================================================
    # SANITIZE
    # =========================================================

    def sanitize_filename(self, text):
        text = re.sub(INVALID_FILENAME_CHARS, "_", text)
        return text.strip(" .")

    def limit_filename(self, filename, limit=180):
        name, ext = os.path.splitext(filename)
        if len(filename) <= limit:
            return filename
        return name[: limit - len(ext)] + ext
