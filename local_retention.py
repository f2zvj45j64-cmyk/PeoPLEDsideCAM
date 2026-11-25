#!/usr/bin/env python3
"""Atomic local image storage with midnight FIFO retention."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional


LogFn = Callable[[str], None]


class LocalRetentionStore:
    """Save locally and remove oldest images in a background midnight job."""

    STATE_NAME = ".retention_state.json"

    def __init__(
        self,
        root: str,
        max_gb: float = 60.0,
        delete_gb: float = 30.0,
        poll_seconds: float = 10.0,
        delete_pause_seconds: float = 0.01,
        log: LogFn = print,
        now_fn: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(float(max_gb) * 1024**3)
        self.delete_bytes = int(float(delete_gb) * 1024**3)
        self.poll_seconds = max(0.1, float(poll_seconds))
        self.delete_pause_seconds = max(0.0, float(delete_pause_seconds))
        self.log = log
        self.now_fn = now_fn
        self.state_path = self.root / self.STATE_NAME
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._total_bytes = self._scan_size()
        self._state = self._load_state()
        self._ensure_schedule_if_full()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="local-retention-worker"
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(0.0, timeout))

    def create_png(self, relative_path: str, png_bytes: bytes) -> Path:
        """Write a complete PNG locally using temp-file + atomic rename."""
        rel = self._safe_relative(relative_path)
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        previous_size = 0
        try:
            if target.exists():
                previous_size = target.stat().st_size
            with open(tmp, "wb") as fh:
                fh.write(png_bytes)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

        with self._lock:
            self._total_bytes = max(
                0, self._total_bytes - previous_size + len(png_bytes)
            )
            self.log(f"LOCAL_CREATED path={rel} bytes={len(png_bytes)}")
            self._ensure_schedule_if_full()
        self._wake.set()
        return target

    def status(self) -> dict:
        with self._lock:
            return {
                "bytes": self._total_bytes,
                "max_bytes": self.max_bytes,
                "delete_bytes": self.delete_bytes,
                "scheduled_for": self._state.get("scheduled_for"),
                "delete_target_bytes": self._state.get("delete_target_bytes"),
                "free_bytes": shutil.disk_usage(self.root).free,
            }

    def run_due_job(self) -> bool:
        """Run a due retention job; public to support deterministic tests."""
        with self._lock:
            scheduled = self._parse_time(self._state.get("scheduled_for"))
            if scheduled is None or self.now_fn() < scheduled:
                return False
            target = self._state.get("delete_target_bytes")
            if target is None:
                target = max(0, self._total_bytes - self.delete_bytes)
                self._state["delete_target_bytes"] = int(target)
                self._save_state()
            target = int(target)

        return self._delete_to_target(target)

    def run_midnight_maintenance(self) -> bool:
        """Independent midnight safety job used by the systemd timer.

        This deliberately rescans disk instead of trusting in-process counters
        or state.  It therefore still works if the GUI crashed, was closed, or
        the Jetson rebooted before midnight.
        """
        with self._lock:
            self._total_bytes = self._scan_size()
            if self._total_bytes < self.max_bytes:
                self.log(
                    f"RETENTION_CHECK_OK current={self._total_bytes} "
                    f"limit={self.max_bytes}"
                )
                return False
            target = max(0, self._total_bytes - self.delete_bytes)
            self._state = {
                "scheduled_for": self.now_fn().isoformat(),
                "triggered_at": self.now_fn().isoformat(),
                "delete_target_bytes": int(target),
            }
            self._save_state()
        return self._delete_to_target(target)

    def _delete_to_target(self, target: int) -> bool:
        self.log(
            f"RETENTION_DELETE_STARTED current={self._total_bytes} target={target}"
        )
        while not self._stop.is_set():
            with self._lock:
                if self._total_bytes <= target:
                    break
            oldest = self._oldest_image()
            if oldest is None:
                break
            try:
                size = oldest.stat().st_size
                rel = oldest.relative_to(self.root)
                oldest.unlink()
                with self._lock:
                    self._total_bytes = max(0, self._total_bytes - size)
                self.log(f"LOCAL_FILE_REMOVED path={rel} bytes={size}")
                self._remove_empty_parents(oldest.parent)
            except FileNotFoundError:
                continue
            except OSError as exc:
                self.log(f"RETENTION_DELETE_FAILED path={oldest} error={exc}")
                self._wake.wait(1.0)
            if self.delete_pause_seconds:
                self._stop.wait(self.delete_pause_seconds)

        with self._lock:
            if self._total_bytes <= target:
                self.log(f"RETENTION_COMPLETE remaining={self._total_bytes}")
                self._state = {}
                self._save_state()
                # New images may have crossed the threshold during deletion.
                self._ensure_schedule_if_full()
                return True
        return False

    def _run(self) -> None:
        while not self._stop.is_set():
            self.run_due_job()
            self._wake.wait(self.poll_seconds)
            self._wake.clear()

    def _ensure_schedule_if_full(self) -> None:
        if self._total_bytes < self.max_bytes or self._state.get("scheduled_for"):
            return
        now = self.now_fn()
        midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self._state = {
            "scheduled_for": midnight.isoformat(),
            "triggered_at": now.isoformat(),
            "delete_target_bytes": None,
        }
        self._save_state()
        self.log(
            f"RETENTION_SCHEDULED current={self._total_bytes} "
            f"limit={self.max_bytes} scheduled_for={midnight.isoformat()}"
        )

    def _scan_size(self) -> int:
        total = 0
        for path in self._images():
            try:
                total += path.stat().st_size
            except OSError:
                pass
        return total

    def _images(self) -> list[Path]:
        images = []
        for path in self.root.rglob("*.png"):
            rel = path.relative_to(self.root)
            if any(part.startswith(".") or part == "_buffer" for part in rel.parts):
                continue
            if path.is_file():
                images.append(path)
        return images

    def _oldest_image(self) -> Optional[Path]:
        images = self._images()
        if not images:
            return None
        return min(images, key=lambda p: (p.stat().st_mtime, str(p)))

    def _load_state(self) -> dict:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_state(self) -> None:
        if not self._state:
            self.state_path.unlink(missing_ok=True)
            return
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.state_path)

    @staticmethod
    def _parse_time(value: object) -> Optional[datetime]:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @staticmethod
    def _safe_relative(path: str) -> Path:
        rel = Path(path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe local path: {path!r}")
        return rel

    def _remove_empty_parents(self, directory: Path) -> None:
        while directory != self.root:
            try:
                directory.rmdir()
            except OSError:
                break
            directory = directory.parent
