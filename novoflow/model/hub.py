"""
Checkpoint download and resolution utilities.
Fetches official pretrained checkpoints from JoelResearch GitHub releases or Hugging Face Hub.
"""

from __future__ import annotations
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path


DEFAULT_RELEASE_REPO = "joelinator/JoelResearch"
DEFAULT_RELEASE_TAG = "v0.3.0"
DEFAULT_FILENAME = "dfm_massive_kb.ckpt"
GITHUB_RELEASE_URL = f"https://github.com/{DEFAULT_RELEASE_REPO}/releases/download/{DEFAULT_RELEASE_TAG}/{DEFAULT_FILENAME}"
HF_BACKUP_URL = f"https://huggingface.co/joelinator/dflow-novo-model/resolve/main/{DEFAULT_FILENAME}"

LOCAL_CANDIDATE_PATHS = [
    os.path.expanduser("~/.cache/novoflow/dfm_massive_kb.ckpt"),
    "/home/joelgedeon_aims_ac_za/dfm-joelresearch/models/dfm_massive_kb.ckpt",
    "/home/joelgedeon_aims_ac_za/checkpoints/dfm_massive_kb.ckpt",
]


def get_default_checkpoint() -> str:
    """
    Resolve or download the default MassIVE-KB pretrained model checkpoint.
    Returns absolute path to the local checkpoint file.
    """
    cache_dir = Path.home() / ".cache" / "novoflow"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / DEFAULT_FILENAME

    if target_path.exists() and target_path.stat().st_size > 10_000_000:
        return str(target_path)

    # 1. Check existing local repository paths
    for p in LOCAL_CANDIDATE_PATHS:
        if os.path.exists(p) and os.path.getsize(p) > 10_000_000:
            try:
                shutil.copyfile(p, str(target_path))
                return str(target_path)
            except Exception:
                return p

    # 2. Try gh release download if available
    try:
        cmd = [
            "gh", "release", "download", DEFAULT_RELEASE_TAG,
            "--repo", DEFAULT_RELEASE_REPO,
            "--pattern", DEFAULT_FILENAME,
            "--dir", str(cache_dir),
            "--clobber",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0 and target_path.exists() and target_path.stat().st_size > 10_000_000:
            return str(target_path)
    except Exception:
        pass

    # 3. Direct HTTP download from GitHub Releases or Hugging Face
    for url in [GITHUB_RELEASE_URL, HF_BACKUP_URL]:
        try:
            print(f"[*] Downloading MassIVE-KB pretrained model from {url}...")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "NovoFlow/0.1.0 (de-novo-sequencing)"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp, open(target_path, "wb") as f:
                shutil.copyfileobj(resp, f)
            if target_path.exists() and target_path.stat().st_size > 10_000_000:
                print(f"[+] Download complete: {target_path} ({target_path.stat().st_size / 1e6:.1f} MB)")
                return str(target_path)
        except Exception as e:
            print(f"[-] Download attempt from {url} failed: {e}")
            if target_path.exists():
                target_path.unlink()

    raise RuntimeError(
        f"Could not automatically download '{DEFAULT_FILENAME}'. "
        f"Please download it manually from {GITHUB_RELEASE_URL} and place it at {target_path}."
    )
