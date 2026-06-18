import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class Dreamless_HF_Downloader:
    BASE_URL = "https://huggingface.co"

    def __init__(
        self,
        repo_id,
        filename,
        token="",
        revision="main",
        save_path=".",
        timeout=60,
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.token = token
        self.revision = revision
        self.save_path = save_path
        self.timeout = timeout
        self.download_url = None
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
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    # =========================================================
    # DETAILS
    # =========================================================

    def details(self):
        """Resolve the final download URL and remote file size via HEAD request."""
        url = f"{self.BASE_URL}/{self.repo_id}/resolve/{self.revision}/{self.filename}"
        self.download_url = url

        req = urllib.request.Request(url, headers=self.headers, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                self.remote_size = int(resp.headers.get("Content-Length", 0))
                self.download_url = resp.geturl()  # follow redirects
        except Exception:
            # fallback: tamanho será determinado na hora do download
            self.remote_size = 0

        return {
            "repo_id": self.repo_id,
            "filename": self.filename,
            "revision": self.revision,
            "download_url": self.download_url,
            "remote_size": self.remote_size,
        }

    # =========================================================
    # DOWNLOAD
    # =========================================================

    def download(self):
        if not self.download_url:
            self.details()

        save_path = str(self.save_path) if self.save_path else "."
        os.makedirs(save_path, exist_ok=True)

        # usa só o basename do filename (pode vir com subpasta ex: "weights/model.safetensors")
        file_basename = os.path.basename(self.filename)
        filepath = os.path.join(save_path, file_basename)

        # -------------------------------------------------
        # Verifica arquivo existente
        # -------------------------------------------------

        if os.path.exists(filepath):
            local_size = os.path.getsize(filepath)
            if self.remote_size > 0 and local_size >= self.remote_size * 0.99:
                print(
                    f"[Dreamless] Using existing file: {file_basename} ({self.format_size(local_size)})"
                )
                return filepath

        print(f"[Dreamless] Multipart download: {self.repo_id}/{self.filename}")

        # -------------------------------------------------
        # Resolve URL final e tamanho total
        # -------------------------------------------------

        try:
            req = urllib.request.Request(
                self.download_url, headers=self.headers, method="HEAD"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                url_final = resp.geturl()
                total_size = int(resp.headers.get("Content-Length", self.remote_size))
        except Exception:
            req = urllib.request.Request(self.download_url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                url_final = resp.geturl()
                total_size = int(resp.headers.get("Content-Length", self.remote_size))

        if total_size <= 0:
            raise RuntimeError("[Dreamless] Failed to determine the size of the file.")

        total_text = self.format_size(total_size)

        # -------------------------------------------------
        # Pre-aloca arquivo
        # -------------------------------------------------

        with open(filepath, "wb") as f:
            f.truncate(total_size)

        # -------------------------------------------------
        # Multipart
        # -------------------------------------------------

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
            part_headers = self.headers.copy()
            part_headers["Range"] = f"bytes={start_byte}-{end_byte}"

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

        # checa erros nas futures
        for f in futures:
            exc = f.exception()
            if exc:
                raise RuntimeError(f"[Dreamless] Download part failed: {exc}")

        print()
        print(f"[Dreamless] {file_basename} downloaded successfully to {filepath}")
        return filepath
