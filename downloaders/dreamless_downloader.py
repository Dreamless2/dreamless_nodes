import os
import requests
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class Dreamless_Downloader:
    API_URL = "https://civitai.com/api/v1"

    def __init__(
        self,
        model_id,
        model_version=None,
        model_types=None,
        token="",
        save_path=".",
        timeout=60,
    ):
        self.model_id = str(model_id)
        self.model_version = model_version
        self.model_types = model_types or []
        self.token = token
        self.save_path = save_path
        self.timeout = timeout
        self.name = None
        self.download_url = None
        self.version = None
        self.remote_size = 0

    @staticmethod
    def format_time(seconds):
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    @staticmethod
    def format_size(size_bytes):
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(size_bytes)
        for unit in units:
            if size < 1024 or unit == units[-1]:
                return f"{size:.2f} {unit}"
            size /= 1024

    @property
    def headers(self):
        headers = {"User-Agent": "Dreamless/1.0"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def details(self):
        if self.model_version:
            url = f"{self.API_URL}/model-versions/{self.model_version}"
            r = requests.get(url, headers=self.headers, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        else:
            url = f"{self.API_URL}/models/{self.model_id}"
            r = requests.get(url, headers=self.headers, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            versions = data.get("modelVersions", [])

            if not versions:
                raise RuntimeError("No versions found.")
            data = versions[0]

        files = data.get("files", [])
        if not files:
            raise RuntimeError("No downloadable files found.")

        selected = None
        for file in files:
            if self.model_types:
                file_type = file.get("type")
                if file_type not in self.model_types:
                    continue
            selected = file
            break

        if not selected:
            selected = files[0]

        self.name = selected["name"]
        self.download_url = selected["downloadUrl"]
        self.version = data["id"]

        size_kb = selected.get("sizeKB", 0)
        self.remote_size = int(size_kb * 1024)

        return selected

    def download(self):
        if not self.download_url or self.name is None:
            self.details()

        assert self.download_url, "Download URL is missing"

        save_path = str(self.save_path) if self.save_path else "."
        os.makedirs(save_path, exist_ok=True)

        assert self.name, "File name is missing"
        filepath = os.path.join(save_path, str(self.name))

        if os.path.exists(filepath):
            local_size = os.path.getsize(filepath)
            if self.remote_size > 0 and local_size >= self.remote_size * 0.99:
                print(
                    f"\33[1m\33[33m[Dreamless] Using existing file:\33[0m {self.name} ({self.format_size(local_size)})"
                )
                return filepath

        print(f"[Dreamless] Multipart download of model {self.name}")

        url_para_resolver = self.download_url
        if self.token:
            if "?" in url_para_resolver:
                url_para_resolver += f"&token={self.token}"
            else:
                url_para_resolver += f"?token={self.token}"

        base_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
        }

        try:
            req = urllib.request.Request(
                url_para_resolver, headers=base_headers, method="HEAD"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                url_final = resp.geturl()
                total_size = int(resp.headers.get("Content-Length", self.remote_size))
        except Exception:
            req = urllib.request.Request(url_para_resolver, headers=base_headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                url_final = resp.geturl()
                total_size = int(resp.headers.get("Content-Length", self.remote_size))

        if total_size <= 0:
            raise RuntimeError("[Dreamless] Failed to determine the size of the file.")

        total_text = self.format_size(total_size)

        with open(filepath, "wb") as f:
            f.truncate(total_size)

        num_threads = 8
        part_size = total_size // num_threads
        ranges = []

        for i in range(num_threads):
            start = i * part_size
            end = total_size - 1 if i == num_threads - 1 else (start + part_size - 1)
            ranges.append((start, end))

        progress = [0] * num_threads
        start_time = time.time()

        def download_part(part_id, start_byte, end_byte):
            part_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "*/*",
                "Range": f"bytes={start_byte}-{end_byte}",
            }

            part_req = urllib.request.Request(url_final, headers=part_headers)
            with urllib.request.urlopen(part_req, timeout=self.timeout) as response:
                with open(filepath, "r+b") as out_file:
                    out_file.seek(start_byte)

                    chunk_size = 1024 * 64  # 64KB
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        progress[part_id] += len(chunk)

        last_percent = -1
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(download_part, idx, r[0], r[1])
                for idx, r in enumerate(ranges)
            ]

            while not all(f.done() for f in futures):
                downloaded = sum(progress)
                percent = int(downloaded * 100 / total_size)

                if percent != last_percent:
                    last_percent = percent
                    elapsed = max(time.time() - start_time, 0.001)
                    speed_bps = downloaded / elapsed
                    remaining = total_size - downloaded
                    eta = remaining / speed_bps if speed_bps > 0 else 0

                    downloaded_text = self.format_size(downloaded)
                    speed_text = f"{self.format_size(speed_bps)}/s"

                    bar_length = 20
                    filled_length = int(bar_length * percent // 100)
                    bar = "█" * filled_length + "░" * (bar_length - filled_length)

                    print(
                        f"\r[Dreamless] [{bar}] {percent}% | Downloaded {downloaded_text} of {total_text} | Speed: {speed_text} | ETA: {self.format_time(eta)}          ",
                        end="",
                        flush=True,
                    )
                time.sleep(0.2)

        print()
        print(f"[Dreamless] Model {self.name} downloaded successfully to {filepath}")
        
        return filepath
        
    