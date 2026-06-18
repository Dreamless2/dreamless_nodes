import os
import re
from pathlib import Path
from dotenv import load_dotenv, set_key

ROOT_PATH = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_PATH / ".env"

if not ENV_PATH.exists():
    ENV_PATH.touch()

load_dotenv(dotenv_path=ENV_PATH)


def get_api_key(provider="civitai") -> str:
    """Reads the API key for the specified provider from the .env file."""
    load_dotenv(dotenv_path=ENV_PATH)
    key_name = f"{provider.upper()}_API_KEY"
    return os.getenv(key_name, "").strip()


def set_api_key(provider: str, key: str):
    """Writes the API key for the specified provider to the .env file."""
    provider = provider.lower().strip()
    key_name = f"{provider.upper()}_API_KEY"
    set_key(str(ENV_PATH), key_name, key.strip())
    os.environ[key_name] = key.strip()


def short_paths_map(paths: list[str]) -> dict:
    """Returns a dictionary with short paths as keys and full paths as values."""
    result = {}
    for path_str in paths:
        p = Path(path_str)
        if p.exists():
            try:
                key = str(Path(*p.parts[-2:])) if len(p.parts) >= 2 else p.name
                result[key] = str(p)
            except:
                result[p.name] = str(p)
    return result


def model_path(filename: str, search_paths: list[str]):
    """Returns the full path to the model file with the specified filename in the search paths."""
    if not filename:
        return None
    filename = filename.lower().strip()
    for base in search_paths:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for file in files:
                if (
                    file.lower() == filename
                    or os.path.splitext(file)[0].lower() == filename
                ):
                    return os.path.join(root, file)
    return None


def sanitize_filename(filename: str) -> str:
    """Sanitizes a filename by replacing invalid characters with underscores and stripping whitespace."""
    invalid = '<>:"/\\|?*'
    for c in invalid:
        filename = filename.replace(c, "_")
    return filename.strip()


def detect_provider(source: str) -> str:
    """Detects the provider type based on the input string format."""
    s = source.lower()
    if "civitai." in s or "urn:air:" in s or re.search(r"^\d+ Bluntly (@\d+)?$", s):
        return "civitai"
    if "huggingface.co" in s:
        return "huggingface"
    return "direct"


def parse_air(air_string: str):
    """
    Parses a CivitAI AIR string in various formats to extract model and version IDs.
        Supported formats:
        1. URL: https://civitai.red/models/443821/...?modelVersionId=2884631
        2. URN: urn:air:sdxl:checkpoint:civitai:443821@2884631
        3. Direct: 443821@2884631 or just the ID
    """
    if not air_string:
        return None, None
    air = air_string.strip()

    if "civitai." in air.lower():
        model_id_match = re.search(r"/models/(\d+)", air)
        version_id_match = re.search(r"modelVersionId=(\d+)", air)

        model_id = int(model_id_match.group(1)) if model_id_match else None
        version_id = int(version_id_match.group(1)) if version_id_match else None
        return model_id, version_id

    if air.startswith("urn:air:"):
        match = re.search(r":(\d+)(?:@(\d+))?$", air)
        if match:
            return int(match.group(1)), int(match.group(2)) if match.group(2) else None

    if "@" in air:
        model, version = air.split("@", 1)
        return int(model) if model.isdigit() else None, int(
            version
        ) if version.isdigit() else None

    return int(air) if air.isdigit() else None, None

def is_hf(value: str) -> bool:
    """Detecta se o valor é uma referência HuggingFace (repo_id/filename)."""
    value = value.strip()
    return (
        value.startswith("hf:")
        or value.startswith("https://huggingface.co/")
        or (value.count("/") >= 2 and not value.startswith("{"))
    )


def parse_hf(value: str):
    """
    Extrai (repo_id, filename, revision) de uma referência HF.

    Formatos aceitos:
        hf:black-forest-labs/FLUX.1-dev/flux1-dev.safetensors
        hf:black-forest-labs/FLUX.1-dev/flux1-dev.safetensors@main
        black-forest-labs/FLUX.1-dev/flux1-dev.safetensors
        https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors
    """
    value = value.strip()

    # URL completa
    if value.startswith("https://huggingface.co/"):
        # https://huggingface.co/{repo_owner}/{repo_name}/resolve/{revision}/{filename}
        parts = value.replace("https://huggingface.co/", "").split("/")
        if len(parts) >= 5 and parts[2] == "resolve":
            repo_id = f"{parts[0]}/{parts[1]}"
            revision = parts[3]
            filename = "/".join(parts[4:])
            return repo_id, filename, revision
        return None, None, "main"

    # Remove prefixo hf:
    if value.startswith("hf:"):
        value = value[3:]

    # revision opcional no final: repo/filename@revision
    revision = "main"
    if "@" in value:
        value, revision = value.rsplit("@", 1)

    # separa repo_id (owner/repo) do filename (pode ter subpastas)
    parts = value.split("/")
    if len(parts) < 3:
        return None, None, revision

    repo_id = f"{parts[0]}/{parts[1]}"
    filename = "/".join(parts[2:])
    return repo_id, filename, revision