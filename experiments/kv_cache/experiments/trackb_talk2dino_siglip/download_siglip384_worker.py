#!/usr/bin/env python3
"""Resumable, pinned Hugging Face download for the paired SigLIP-384 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download


REPO_ID = "timm/ViT-SO400M-14-SigLIP-384"
REVISION = "ac16108d567c4389e6cd2b11c9b8585f7474435b"
EXPECTED_SHA256 = "97faeafa0576c1ba4c92bae739ca06ef5f5d0c2db92f86a6086653d82a262043"
EXPECTED_SIZE = 3_511_918_424
SMALL_FILES = (
    "open_clip_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def update_status(path: Path, **values) -> None:
    current = {}
    if path.is_file():
        try:
            current = json.loads(path.read_text())
        except Exception:
            pass
    current.update({
        "repo_id": REPO_ID,
        "revision": REVISION,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        **values,
    })
    atomic_json(path, current)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_remote_weight_metadata() -> dict:
    url = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/open_clip_model.safetensors"
    response = requests.head(url, allow_redirects=False, timeout=30)
    response.raise_for_status()
    headers = response.headers
    remote = {
        "repo_commit": headers.get("X-Repo-Commit"),
        "linked_size": int(headers.get("X-Linked-Size", "0")),
        "linked_etag_sha256": headers.get("X-Linked-ETag", "").strip('"'),
        "xet_hash": headers.get("X-Xet-Hash"),
    }
    if remote["repo_commit"] != REVISION:
        raise RuntimeError(f"Remote commit mismatch: {remote}")
    if remote["linked_size"] != EXPECTED_SIZE:
        raise RuntimeError(f"Remote size mismatch: {remote}")
    if remote["linked_etag_sha256"] != EXPECTED_SHA256:
        raise RuntimeError(f"Official revision SHA metadata mismatch: {remote}")
    return remote


def check_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    model = config["model_cfg"]
    text = model["text_cfg"]
    observed = {
        "context_length": int(text["context_length"]),
        "embed_dim": int(model["embed_dim"]),
        "text_width": int(text["width"]),
        "layers": int(text["layers"]),
        "heads": int(text["heads"]),
        "clean": text.get("tokenizer_kwargs", {}).get("clean"),
        "tokenizer_name": text.get("hf_tokenizer_name"),
    }
    expected = {
        "context_length": 64, "embed_dim": 1152, "text_width": 1152,
        "layers": 27, "heads": 16, "clean": "canonicalize",
    }
    for key, value in expected.items():
        if observed[key] != value:
            raise RuntimeError(f"Unexpected checkpoint config value: {key}={observed[key]!r}, expected {value!r}")
    return observed


def download_one(filename: str, destination: Path, status_path: Path, retries: int) -> Path:
    final_path = destination / filename
    if final_path.is_file():
        if filename == "open_clip_model.safetensors":
            size = final_path.stat().st_size
            digest = sha256(final_path) if size == EXPECTED_SIZE else None
            if size == EXPECTED_SIZE and digest == EXPECTED_SHA256:
                logging.info("Reusing already verified %s", final_path)
                return final_path
            raise FileExistsError(f"Existing weight file is not the expected artifact; preserving it: {final_path}")

    for attempt in range(1, retries + 1):
        try:
            update_status(status_path, state="downloading", current_file=filename,
                          attempt=attempt, max_attempts=retries,
                          present_bytes=final_path.stat().st_size if final_path.exists() else 0)
            logging.info("Downloading %s (attempt %d/%d); Hub local-dir download resumes partial files", filename, attempt, retries)
            result = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                revision=REVISION,
                local_dir=str(destination),
                local_dir_use_symlinks=False,
            )
            local_path = Path(result)
            if not local_path.is_file():
                raise FileNotFoundError(f"Hub returned missing local file {local_path}")
            logging.info("Downloaded %s bytes=%d", filename, local_path.stat().st_size)
            return local_path
        except Exception as exc:
            if attempt == retries:
                raise
            delay = min(180, 5 * (2 ** (attempt - 1)))
            logging.exception("Download failed for %s; retry in %ds (partial file retained)", filename, delay)
            update_status(status_path, state="retrying", current_file=filename,
                          attempt=attempt, next_retry_seconds=delay,
                          last_error=f"{type(exc).__name__}: {exc}")
            time.sleep(delay)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    update_status(args.status, state="pending", destination=str(args.destination), files=list(SMALL_FILES) + ["open_clip_model.safetensors"])
    try:
        remote = verify_remote_weight_metadata()
        logging.info("Pinned official revision metadata verified: %s", remote)
        update_status(args.status, state="downloading_metadata", remote_metadata=remote)
        for filename in SMALL_FILES:
            download_one(filename, args.destination, args.status, args.retries)
        for filename in SMALL_FILES:
            json.loads((args.destination / filename).read_text())
        config = check_config(args.destination / "open_clip_config.json")
        update_status(args.status, state="downloading_weights", config_validation=config)
        weights = download_one("open_clip_model.safetensors", args.destination, args.status, args.retries)
        update_status(args.status, state="verifying", weight_path=str(weights), weight_size=weights.stat().st_size)
        if weights.stat().st_size != EXPECTED_SIZE:
            raise RuntimeError(f"Weight size mismatch: {weights.stat().st_size} != {EXPECTED_SIZE}")
        actual_sha = sha256(weights)
        if actual_sha != EXPECTED_SHA256:
            raise RuntimeError(f"Weight SHA256 mismatch: {actual_sha}")
        downloaded = {name: str(args.destination / name) for name in (*SMALL_FILES, "open_clip_model.safetensors")}
        absent = [name for name, path in downloaded.items() if not Path(path).is_file()]
        if absent:
            raise FileNotFoundError(f"Downloaded file set is incomplete: {absent}")
        update_status(args.status, state="complete", downloaded_paths=downloaded,
                      weight_sha256=actual_sha, weight_size=weights.stat().st_size,
                      config_validation=config, remote_metadata=remote)
        logging.info("COMPLETE: %s SHA256=%s", weights, actual_sha)
    except Exception as exc:
        update_status(args.status, state="failed", last_error=f"{type(exc).__name__}: {exc}")
        logging.exception("Checkpoint download/verification failed")
        raise


if __name__ == "__main__":
    main()
