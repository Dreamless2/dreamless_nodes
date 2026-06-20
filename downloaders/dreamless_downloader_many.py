import os
import sys
import time
import threading
import urllib.request
import requests
from concurrent.futures import ThreadPoolExecutor


class _MultiProgressRenderer:
    """
    Renderiza N barras de progresso independentes no terminal usando
    cursor ANSI. Cada slot tem sua própria linha e é atualizado
    individualmente sem sobrescrever as outras.
    """

    def __init__(self, slots: list):
        self._lock = threading.Lock()
        self._slots = list(slots)
        self._lines = [""] * len(slots)
        self._started = False

    def _reserve(self):
        for _ in self._slots:
            sys.stdout.write("\n")
        sys.stdout.flush()
        self._started = True

    def update(self, slot_idx: int, text: str):
        with self._lock:
            if not self._started:
                self._reserve()

            n = len(self._slots)
            lines_up = n - slot_idx

            sys.stdout.write(f"\033[{lines_up}A")  # sobe até a linha do slot
            sys.stdout.write("\r\033[K")            # limpa a linha
            sys.stdout.write(text)
            sys.stdout.write(f"\033[{lines_up}B")  # desce de volta
            sys.stdout.flush()
            self._lines[slot_idx] = text

    def finish(self, slot_idx: int, text: str):
        self.update(slot_idx, text)

    def close(self):
        with self._lock:
            sys.stdout.write("\n")
            sys.stdout.flush()


class Dreamless_Downloader_Many:
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

    @staticmethod
    def _make_bar(percent, length=20):
        filled = int(length * percent // 100)
        return "█" * filled + "░" * (length - filled)

    @property
    def headers(self):
        h = {"User-Agent": "Dreamless/1.0"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

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
                if file.get("type") not in self.model_types:
                    continue
            selected = file
            break

        if not selected:
            selected = files[0]

        self.name = selected["name"]
        self.download_url = selected["downloadUrl"]
        self.version = data["id"]
        self.remote_size = int(selected.get("sizeKB", 0) * 1024)

        return selected

    def download(self, renderer=None, slot_idx: int = 0):
        if not self.download_url or self.name is None:
            self.details()

        assert self.download_url, "Download URL is missing"
        assert self.name, "File name is missing"

        save_path = str(self.save_path) if self.save_path else "."
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, str(self.name))

        label = f"\33[1m{self.name}\33[0m"

        # ── Arquivo já existe ─────────────────────────────────────────────────
        if os.path.exists(filepath):
            local_size = os.path.getsize(filepath)
            if self.remote_size > 0 and local_size >= self.remote_size * 0.99:
                msg = f"\33[1m\33[33m[Dreamless] Using existing file:\33[0m {label} ({self.format_size(local_size)})"
                if renderer:
                    renderer.finish(slot_idx, msg)
                else:
                    print(msg)
                return filepath

        # ── Resolve URL final e tamanho real via Content-Length ───────────────
        url_para_resolver = self.download_url
        if self.token:
            sep = "&" if "?" in url_para_resolver else "?"
            url_para_resolver += f"{sep}token={self.token}"

        base_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
        }

        try:
            req = urllib.request.Request(url_para_resolver, headers=base_headers, method="HEAD")
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

        # ── Pré-aloca o arquivo ───────────────────────────────────────────────
        with open(filepath, "wb") as f:
            f.truncate(total_size)

        # ── Divide em partes ──────────────────────────────────────────────────
        num_threads = 8
        part_size = total_size // num_threads
        ranges = [
            (
                i * part_size,
                total_size - 1 if i == num_threads - 1 else (i + 1) * part_size - 1,
            )
            for i in range(num_threads)
        ]

        progress = [0] * num_threads
        start_time = time.time()

        def download_part(part_id, start_byte, end_byte):
            part_headers = {
                **base_headers,
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

        # ── Loop de progresso ─────────────────────────────────────────────────
        last_percent = -1
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(download_part, idx, r[0], r[1])
                for idx, r in enumerate(ranges)
            ]

            while not all(f.done() for f in futures):
                downloaded = sum(progress)
                percent = min(int(downloaded * 100 / total_size), 100)

                if percent != last_percent:
                    last_percent = percent
                    elapsed = max(time.time() - start_time, 0.001)
                    speed_bps = downloaded / elapsed
                    remaining = total_size - downloaded
                    eta = remaining / speed_bps if speed_bps > 0 else 0

                    bar = self._make_bar(percent)
                    line = (
                        f"[Dreamless] {label} "
                        f"[{bar}] {percent}% | "
                        f"{self.format_size(downloaded)} of {total_text} | "
                        f"{self.format_size(speed_bps)}/s | "
                        f"EETA: {self.format_time(eta)}          "
                    )

                    if renderer:
                        renderer.update(slot_idx, line)
                    else:
                        print(f"\r{line}", end="", flush=True)

                time.sleep(0.2)

        # ── Finaliza ──────────────────────────────────────────────────────────
        done_line = f"\33[1m\33[32m[Dreamless] ✓ {label}\33[0m downloaded to {filepath}"
        if renderer:
            renderer.finish(slot_idx, done_line)
        else:
            print()
            print(done_line)

        return filepath

    @staticmethod
    def download_many(downloaders, max_workers=2, stagger_seconds=3):
        """
        Baixa múltiplos modelos em paralelo com barras de progresso individuais.

        downloaders:     lista de instâncias Dreamless_Downloader já configuradas
        max_workers:     downloads simultâneos (cada um usa 8 threads internamente)
        stagger_seconds: delay entre inícios para não estressar a API
        """
        # Busca metadados de todos antes de reservar as linhas do terminal
        for dl in downloaders:
            if dl.name is None:
                dl.details()

        renderer = _MultiProgressRenderer([dl.name for dl in downloaders])
        results = {}

        def run(args):
            idx, dl = args
            time.sleep(idx * stagger_seconds)
            try:
                path = dl.download(renderer=renderer, slot_idx=idx)
                return dl.model_id, path
            except Exception as e:
                renderer.finish(idx, f"\33[1m\33[31m[Dreamless] ✗ {dl.name}\33[0m — {e}")
                return dl.model_id, e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run, (idx, dl)): dl
                for idx, dl in enumerate(downloaders)
            }
            for future in futures:
                model_id, result = future.result()
                results[model_id] = result

        renderer.close()
        return results