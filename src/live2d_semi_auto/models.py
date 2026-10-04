"""Explicit model acquisition with TLS and a pinned artifact checksum."""

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request


LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
LAMA_SHA256 = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"


def verify_lama(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise ValueError("LaMaモデルがありません。READMEの明示ダウンロード手順を実行してください。")
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != LAMA_SHA256:
        raise ValueError("LaMaモデルのチェックサムが一致しません。再取得してください。")
    return path


def download_lama(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return verify_lama(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            with urllib.request.urlopen(LAMA_URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
        verify_lama(temporary)
        # Same-directory hard link publishes complete bytes without overwriting.
        os.link(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="明示的にLaMaモデルをダウンロード（約196MiB）")
    parser.add_argument("command", choices=["download-lama"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(download_lama(args.output))


if __name__ == "__main__":
    main()
