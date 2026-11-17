#!/usr/bin/env python3
"""Restartable COCO 2014 downloader/extractor for the Talk2DINO preparation.

Transfers use the official COCO S3 bucket. The canonical images.cocodataset.org
hostname currently presents an S3 certificate that does not match that host;
this script keeps TLS verification enabled by connecting to s3.amazonaws.com and
using the official COCO bucket host header. No mirror or insecure TLS mode is used.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

COCO_HOST = "images.cocodataset.org"
AWS_BASE = "https://s3.us-east-1.amazonaws.com"
FILES = [
    {
        "key": "annotations",
        "path": "annotations/annotations_trainval2014.zip",
        "canonical_url": "https://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        "size": 252_872_794,
        "archive": "annotations_trainval2014.zip",
        "extract_dir": "annotations2014",
        "members": ["annotations/captions_train2014.json", "annotations/captions_val2014.json"],
    },
    {
        "key": "train2014",
        "path": "zips/train2014.zip",
        "canonical_url": "https://images.cocodataset.org/zips/train2014.zip",
        "size": 13_510_573_713,
        "archive": "train2014.zip",
        "extract_dir": "train2014",
        "prefix": "train2014/",
    },
    {
        "key": "val2014",
        "path": "zips/val2014.zip",
        "canonical_url": "https://images.cocodataset.org/zips/val2014.zip",
        "size": 6_645_013_297,
        "archive": "val2014.zip",
        "extract_dir": "val2014",
        "prefix": "val2014/",
    },
]
APPROX_PEAK_BYTES = 50_000_000_000  # archives + extracted images + compact feature caches + margin
STOP = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_member(name: str) -> bool:
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts


class Worker:
    def __init__(self, root: Path, status_path: Path):
        self.root = root.resolve()
        self.status_path = status_path.resolve()
        self.archives = self.root / "archives"
        self.state: dict[str, Any] = {
            "schema_version": 1,
            "experiment": "trackb_talk2dino_siglip_coco2014_preparation",
            "status": "pending",
            "pid": os.getpid(),
            "host": os.uname().nodename,
            "dataset_root": str(self.root),
            "status_path": str(self.status_path),
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "transport": {
                "official_host": COCO_HOST,
                "tls_verified_endpoint": AWS_BASE,
                "host_header": COCO_HOST,
                "canonical_urls": {f["key"]: f["canonical_url"] for f in FILES},
                "no_mirror": True,
                "tls_verification": "enabled",
            },
            "required_peak_bytes_estimate": APPROX_PEAK_BYTES,
            "files": {},
            "error": None,
        }
        if self.status_path.exists():
            try:
                old = json.loads(self.status_path.read_text())
                self.state.update(old)
                self.state["pid"] = os.getpid()
                self.state["updated_at_utc"] = utc_now()
                self.state["error"] = None
            except (OSError, json.JSONDecodeError):
                pass
        self.logger = logging.getLogger("talk2dino-coco")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S"))
        self.logger.addHandler(h)
        self.root.mkdir(parents=True, exist_ok=True)
        self.archives.mkdir(parents=True, exist_ok=True)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self._status_lock = threading.RLock()
        self.write_status()

    def write_status(self, **extra: Any) -> None:
        with self._status_lock:
            self.state.update(extra)
            self.state["updated_at_utc"] = utc_now()
            atomic_json(self.status_path, self.state)

    def set_status(self, status: str, **extra: Any) -> None:
        self.state["status"] = status
        self.write_status(**extra)
        self.logger.info("STATUS %s %s", status, json.dumps(extra, sort_keys=True))

    def set_file(self, key: str, **values: Any) -> None:
        with self._status_lock:
            row = self.state.setdefault("files", {}).setdefault(key, {})
            row.update(values)
            if values.get("status") in {"downloading", "verified", "extracting", "complete"}:
                row_status = values["status"]
                if row_status == "verified":
                    self.state["status"] = "verifying"
                elif row_status == "complete":
                    self.state["status"] = "extracting"
                else:
                    self.state["status"] = row_status
            row["updated_at_utc"] = utc_now()
            self.write_status()

    def remote_metadata(self, spec: dict[str, Any]) -> tuple[str | None, int]:
        req = urllib.request.Request(
            AWS_BASE + "/" + spec["path"],
            headers={"Host": COCO_HOST, "User-Agent": "TrackingKVinOpenVLA-Talk2DINO-COCO/1.0"},
            method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            length = int(response.headers.get("Content-Length", "-1"))
            etag = response.headers.get("ETag")
            if length != spec["size"]:
                raise RuntimeError(f"Official object size changed for {spec['key']}: expected {spec['size']}, got {length}.")
            return etag, length

    def download(self, spec: dict[str, Any]) -> Path:
        final = self.archives / spec["archive"]
        part = final.with_suffix(final.suffix + ".part")
        if final.exists() and final.stat().st_size == spec["size"]:
            self.logger.info("Archive already has expected size; reusing %s", final)
            return final
        if final.exists():
            raise RuntimeError(f"Unverified archive exists at {final}; leaving it untouched.")

        etag, expected = self.remote_metadata(spec)
        self.set_file(spec["key"], status="downloading", url=spec["canonical_url"], transport_url=AWS_BASE + "/" + spec["path"], expected_bytes=expected, etag=etag, part_path=str(part))
        workers = 8
        prefix_bytes = part.stat().st_size if part.exists() else 0
        if prefix_bytes > expected:
            raise RuntimeError(f"Partial file is larger than expected: {part}")
        if prefix_bytes == expected:
            os.replace(part, final)
            return final
        remaining = expected - prefix_bytes
        segment_size = (remaining + workers - 1) // workers
        segments: list[tuple[int, int, Path]] = []
        for index in range(workers):
            start = prefix_bytes + index * segment_size
            if start >= expected:
                break
            end = min(expected - 1, start + segment_size - 1)
            segments.append((start, end, Path(str(part) + f".seg{index:02d}")))
        self.set_file(spec["key"], status="downloading", workers=workers, resume_prefix_bytes=prefix_bytes, segment_files=[str(p) for _, _, p in segments])

        def report_progress() -> None:
            total = prefix_bytes
            for start, end, path in segments:
                if path.exists():
                    total += min(path.stat().st_size, end - start + 1)
            self.set_file(spec["key"], status="downloading", downloaded_bytes=total, fraction=round(total / expected, 5))

        def fetch_segment(start: int, end: int, path: Path) -> None:
            length = end - start + 1
            failures = 0
            while not STOP:
                offset = path.stat().st_size if path.exists() else 0
                if offset > length:
                    raise RuntimeError(f"Partial range is larger than expected: {path}")
                if offset == length:
                    return
                absolute_start = start + offset
                headers = {
                    "Host": COCO_HOST,
                    "User-Agent": "TrackingKVinOpenVLA-Talk2DINO-COCO/1.0",
                    "Range": f"bytes={absolute_start}-{end}",
                }
                if etag:
                    headers["If-Range"] = etag
                request = urllib.request.Request(AWS_BASE + "/" + spec["path"], headers=headers)
                try:
                    with urllib.request.urlopen(request, timeout=90) as response:
                        status = getattr(response, "status", response.getcode())
                        content_range = response.headers.get("Content-Range", "")
                        response_etag = response.headers.get("ETag")
                        if status != 206 or not content_range.startswith(f"bytes {absolute_start}-"):
                            raise RuntimeError(f"Range request was not honored for {spec['key']}: HTTP {status}, {content_range!r}")
                        if etag and response_etag != etag:
                            raise RuntimeError(f"ETag changed while downloading {spec['key']}: {etag!r} -> {response_etag!r}")
                        mode = "ab" if offset else "wb"
                        current = offset
                        last_report = time.monotonic()
                        with path.open(mode) as out:
                            while not STOP:
                                block = response.read(2 * 1024 * 1024)
                                if not block:
                                    break
                                out.write(block)
                                current += len(block)
                                if time.monotonic() - last_report >= 10:
                                    out.flush()
                                    report_progress()
                                    last_report = time.monotonic()
                            out.flush()
                            os.fsync(out.fileno())
                    if STOP:
                        return
                    if path.stat().st_size == length:
                        self.logger.info("Range %s [%d:%d] complete", spec["key"], start, end)
                        report_progress()
                        return
                    failures += 1
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                    failures += 1
                    self.logger.warning("Range error %s [%d:%d] attempt %d: %s", spec["key"], start, end, failures, type(exc).__name__ + ": " + str(exc))
                if failures >= 8:
                    delay = min(900, 30 * (2 ** min(failures - 8, 5)))
                    self.set_status("blocked", blocked_file=spec["key"], retry_in_seconds=delay, last_error="range transfer interrupted; completed ranges retained")
                    time.sleep(delay)
                    failures = 0
                    self.set_status("downloading", retrying_file=spec["key"])
            if STOP:
                return

        self.logger.info("Download %s using %d resumable byte ranges; existing prefix %.2f MiB", spec["key"], len(segments), prefix_bytes / 2**20)
        with ThreadPoolExecutor(max_workers=min(workers, len(segments))) as pool:
            futures = [pool.submit(fetch_segment, start, end, path) for start, end, path in segments]
            for future in futures:
                future.result()
        if STOP:
            report_progress()
            raise KeyboardInterrupt
        sizes = [path.stat().st_size if path.exists() else 0 for _, _, path in segments]
        required = [end - start + 1 for start, end, _ in segments]
        if sizes != required:
            raise RuntimeError(f"One or more download ranges are incomplete for {spec['key']}: {sizes} != {required}")

        assembled = Path(str(part) + ".assemble")
        assembled.unlink(missing_ok=True)
        with assembled.open("wb") as out:
            if part.exists():
                with part.open("rb") as prefix:
                    shutil.copyfileobj(prefix, out, length=8 * 1024 * 1024)
            for _, _, path in segments:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        if assembled.stat().st_size != expected:
            raise RuntimeError(f"Assembled archive size mismatch for {spec['key']}.")
        os.replace(assembled, final)
        part.unlink(missing_ok=True)
        for _, _, path in segments:
            path.unlink(missing_ok=True)
        self.set_file(spec["key"], downloaded_bytes=expected, fraction=1.0, range_count=len(segments))
        return final

    def verify_zip(self, spec: dict[str, Any], archive: Path) -> dict[str, Any]:
        self.set_status("verifying", verifying_file=spec["key"])
        h = hashlib.sha256()
        with archive.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"Archive CRC failure at member {bad!r} in {archive.name}")
            if "members" in spec:
                missing = sorted(set(spec["members"]) - set(names))
                if missing:
                    raise RuntimeError(f"Required annotation members missing: {missing}")
            else:
                jpgs = [name for name in names if name.startswith(spec["prefix"]) and name.lower().endswith(".jpg")]
                if not jpgs:
                    raise RuntimeError(f"No expected JPEG members found under {spec['prefix']!r}.")
                spec["jpg_members"] = jpgs
            manifest = {"archive": str(archive), "size_bytes": archive.stat().st_size, "sha256": h.hexdigest(), "etag": self.state.get("files", {}).get(spec["key"], {}).get("etag"), "member_count": len(names), "verified_zip_crc": True}
        self.set_file(spec["key"], status="verified", **manifest)
        self.logger.info("Verified %s: %s", spec["key"], json.dumps({k: v for k, v in manifest.items() if k != "sha256"}, sort_keys=True))
        return manifest

    def safe_extract(self, spec: dict[str, Any], archive: Path, manifest: dict[str, Any]) -> None:
        self.set_status("extracting", extracting_file=spec["key"])
        destination = self.root / spec["extract_dir"]
        if "members" in spec:
            complete_paths = [destination / PurePosixPath(name).name for name in spec["members"]]
            if all(p.is_file() and p.stat().st_size > 0 for p in complete_paths):
                self.set_file(spec["key"], extraction_status="reused_existing", extracted_paths=[str(p) for p in complete_paths])
                return
        else:
            members = spec["jpg_members"]
            if destination.is_dir():
                if all((destination / PurePosixPath(n).name).is_file() for n in members):
                    self.set_file(spec["key"], extraction_status="reused_existing", image_count=len(members), extracted_dir=str(destination))
                    return
                raise RuntimeError(f"Destination exists but is incomplete; refusing to overwrite: {destination}")

        staging = self.root / f".{spec['extract_dir']}.talk2dino-staging"
        marker = staging / ".talk2dino_owned_staging.json"
        if staging.exists():
            if not marker.is_file():
                raise RuntimeError(f"Refusing to touch unmarked staging directory {staging}")
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        marker.write_text(json.dumps({"archive_sha256": manifest["sha256"], "key": spec["key"]}) + "\n")
        self.logger.info("Extracting %s into staging (%s members)", spec["key"], manifest["member_count"])
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not safe_member(info.filename):
                    raise RuntimeError(f"Unsafe archive path {info.filename!r}")
            zf.extractall(staging)

        if "members" in spec:
            source_root = staging / "annotations"
            destination.mkdir(parents=True, exist_ok=False)
            written: list[str] = []
            for member in spec["members"]:
                target = destination / PurePosixPath(member).name
                if target.exists():
                    raise RuntimeError(f"Refusing to overwrite annotation file {target}")
                shutil.copy2(source_root / PurePosixPath(member).name, target)
                written.append(str(target))
            atomic_json(destination / "talk2dino_manifest.json", {**manifest, "files": written})
            shutil.rmtree(staging)
        else:
            staged_images = staging / spec["extract_dir"]
            if not staged_images.is_dir() or len([p for p in staged_images.iterdir() if p.suffix.lower() == ".jpg"]) != len(spec["jpg_members"]):
                raise RuntimeError(f"Extracted image count mismatch for {spec['key']}.")
            atomic_json(staged_images / ".talk2dino_manifest.json", {**manifest, "image_count": len(spec["jpg_members"])})
            if destination.exists():
                raise RuntimeError(f"Refusing to overwrite dataset destination {destination}")
            os.replace(staged_images, destination)
            shutil.rmtree(staging)
            written = [str(destination)]
        self.set_file(spec["key"], extraction_status="complete", extracted_paths=written, image_count=len(spec.get("jpg_members", [])) or None)

    def run(self) -> None:
        usage = shutil.disk_usage(self.root)
        self.state["disk"] = {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free, "required_peak_bytes_estimate": APPROX_PEAK_BYTES}
        self.write_status()
        if usage.free < APPROX_PEAK_BYTES:
            self.set_status("blocked", reason="insufficient persistent storage", free_bytes=usage.free, required_bytes=APPROX_PEAK_BYTES)
            return
        for spec in FILES:
            if STOP:
                self.set_status("pending", reason="stopped by signal", next_file=spec["key"])
                return
            archive = self.download(spec)
            manifest = self.verify_zip(spec, archive)
            self.safe_extract(spec, archive, manifest)
            self.set_file(spec["key"], status="complete")
        ann_dir = self.root / "annotations2014"
        expected = [ann_dir / "captions_train2014.json", ann_dir / "captions_val2014.json"]
        if not all(p.is_file() for p in expected):
            raise RuntimeError("Final caption annotation files are missing.")
        self.set_status("complete", completed_at_utc=utc_now(), annotation_files=[str(p) for p in expected], note="Dataset downloaded, archive CRC-checked and extracted. Feature extraction/training not started.")
        while not STOP:
            time.sleep(60)
            self.write_status(heartbeat_at_utc=utc_now())
        self.set_status("complete", worker_stopped_at_utc=utc_now(), reason="requested shutdown after completion")


def handle_stop(signum: int, _frame: Any) -> None:
    global STOP
    STOP = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_handler = logging.FileHandler(args.log, encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S"))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(log_handler)
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    lock_path = args.dataset_root / ".talk2dino_prepare.lock"
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another COCO preparation worker holds the dataset lock.", flush=True)
        return 3
    worker = Worker(args.dataset_root, args.status)
    try:
        worker.run()
        return 0
    except KeyboardInterrupt:
        worker.set_status("pending", reason="stopped; partial download retained for restart")
        return 130
    except Exception as exc:
        worker.set_status("failed", error=f"{type(exc).__name__}: {exc}")
        worker.logger.exception("Worker failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
