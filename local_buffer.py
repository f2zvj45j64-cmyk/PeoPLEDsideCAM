#!/usr/bin/env python3
"""Durable local image buffer with verified background network archiving."""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional


LogFn = Callable[[str], None]


class LocalBuffer:
    """Persist files locally, then archive them FIFO without blocking capture."""

    STATES = ("PENDING", "TRANSFERRING", "CONFIRMED")

    def __init__(
        self,
        root: str,
        network_root: str,
        network_mount: str,
        max_gb: float = 50.0,
        min_free_gb: float = 20.0,
        verify_checksum: bool = False,
        poll_seconds: float = 5.0,
        status_seconds: float = 300.0,
        log: LogFn = print,
        require_network_mount: bool = True,
    ) -> None:
        self.root = Path(root)
        self.network_root = Path(network_root)
        self.network_mount = Path(network_mount)
        self.max_bytes = int(max_gb * 1024**3)
        self.min_free_bytes = int(min_free_gb * 1024**3)
        self.verify_checksum = bool(verify_checksum)
        self.poll_seconds = max(0.1, float(poll_seconds))
        self.status_seconds = max(1.0, float(status_seconds))
        self.log = log
        self.require_network_mount = require_network_mount
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_log: dict[str, float] = {}
        self._last_status = 0.0
        self._count_lock = threading.Lock()
        self._buffer_bytes = 0
        self._pending_count = 0
        for state in self.STATES:
            (self.root / state).mkdir(parents=True, exist_ok=True)
        self._refresh_counters()

    @property
    def pending_dir(self) -> Path:
        return self.root / "PENDING"

    @property
    def transferring_dir(self) -> Path:
        return self.root / "TRANSFERRING"

    @property
    def confirmed_dir(self) -> Path:
        return self.root / "CONFIRMED"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.recover()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="network-archive-worker"
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(0.0, timeout))

    def create_png(self, relative_path: str, png_bytes: bytes) -> Path:
        """Atomically publish a fully encoded PNG as PENDING."""
        rel = self._safe_relative(relative_path)
        target = self.pending_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
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
        self.log(f"BUFFER_CREATED path={rel} bytes={len(png_bytes)}")
        with self._count_lock:
            self._buffer_bytes += len(png_bytes)
            self._pending_count += 1
        self._check_limits(full_scan=False)
        self._wake.set()
        return target

    def recover(self) -> None:
        """An interrupted transfer is never assumed successful; requeue it."""
        for src in self._files(self.transferring_dir):
            rel = src.relative_to(self.transferring_dir)
            dst = self.pending_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                self._rate_log("TRANSFER_FAILED", f"recovery_collision={rel}")
                continue
            os.replace(src, dst)
        # CONFIRMED means verification and state transition both completed.
        for confirmed in self._files(self.confirmed_dir):
            confirmed.unlink(missing_ok=True)
        self._refresh_counters()

    def process_one(self) -> bool:
        pending = self._oldest_pending()
        if pending is None:
            return False
        if not self.network_available():
            self._rate_log("NETWORK_UNAVAILABLE", f"mount={self.network_mount}")
            return False

        rel = pending.relative_to(self.pending_dir)
        transferring = self.transferring_dir / rel
        transferring.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(pending, transferring)
        except FileNotFoundError:
            return True

        destination = self.network_root / rel
        try:
            self.log(f"TRANSFER_STARTED path={rel}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            expected_size = transferring.stat().st_size

            if destination.exists():
                if self._verified(transferring, destination, expected_size):
                    self._confirm_and_remove(transferring, rel)
                    return True
                raise IOError("destination exists but verification does not match")

            part = destination.with_name(f".{destination.name}.part")
            try:
                part.unlink(missing_ok=True)
                with open(transferring, "rb") as src, open(part, "xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                    dst.flush()
                    os.fsync(dst.fileno())
                os.replace(part, destination)
            finally:
                try:
                    part.unlink(missing_ok=True)
                except OSError:
                    pass

            if not self._verified(transferring, destination, expected_size):
                raise IOError("destination verification failed")
            self._confirm_and_remove(transferring, rel)
            return True
        except Exception as exc:
            # Requeue for retry. If power fails first, recover() does the same.
            pending_again = self.pending_dir / rel
            pending_again.parent.mkdir(parents=True, exist_ok=True)
            try:
                if transferring.exists() and not pending_again.exists():
                    os.replace(transferring, pending_again)
            except OSError:
                pass
            self._rate_log("TRANSFER_FAILED", f"path={rel} error={exc}")
            return False

    def network_available(self) -> bool:
        if self.require_network_mount and not os.path.ismount(self.network_mount):
            return False
        return self.network_mount.is_dir() and os.access(self.network_mount, os.W_OK)

    def stats(self) -> dict:
        files = self._files(self.pending_dir) + self._files(self.transferring_dir)
        sizes = []
        for path in files:
            try:
                sizes.append((path.stat().st_mtime, path.stat().st_size, path))
            except OSError:
                pass
        free = shutil.disk_usage(self.root).free
        oldest = min(sizes, default=None, key=lambda item: item[0])
        return {
            "pending_count": len(sizes),
            "pending_bytes": sum(item[1] for item in sizes),
            "free_bytes": free,
            "oldest": str(oldest[2]) if oldest else "none",
            "network": self.network_available(),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self._check_limits(full_scan=False)
            now = time.monotonic()
            if now - self._last_status >= self.status_seconds:
                s = self.stats()
                self.log(
                    "BUFFER_STATUS "
                    f"pending={s['pending_count']} bytes={s['pending_bytes']} "
                    f"free={s['free_bytes']} oldest={s['oldest']} "
                    f"network={'UP' if s['network'] else 'DOWN'}"
                )
                self._last_status = now
            worked = self.process_one()
            if not worked:
                self._wake.wait(self.poll_seconds)
                self._wake.clear()

    def _check_limits(self, full_scan: bool = True) -> None:
        if full_scan:
            self._refresh_counters()
        with self._count_lock:
            pending_bytes = self._buffer_bytes
        free_bytes = shutil.disk_usage(self.root).free
        if pending_bytes >= self.max_bytes:
            self._rate_log(
                "BUFFER_LIMIT_WARNING",
                f"bytes={pending_bytes} limit={self.max_bytes}",
                interval=60.0,
            )
            self._wake.set()
        if free_bytes <= self.min_free_bytes:
            self._rate_log(
                "LOW_DISK_SPACE",
                f"free={free_bytes} minimum={self.min_free_bytes}",
                interval=60.0,
            )
            self._wake.set()

    def _confirm_and_remove(self, transferring: Path, rel: Path) -> None:
        size = transferring.stat().st_size
        confirmed = self.confirmed_dir / rel
        confirmed.parent.mkdir(parents=True, exist_ok=True)
        os.replace(transferring, confirmed)
        self.log(f"TRANSFER_CONFIRMED path={rel}")
        confirmed.unlink()
        with self._count_lock:
            self._buffer_bytes = max(0, self._buffer_bytes - size)
            self._pending_count = max(0, self._pending_count - 1)
        self.log(f"LOCAL_COPY_REMOVED path={rel}")

    def _verified(self, source: Path, destination: Path, size: int) -> bool:
        try:
            if not destination.is_file() or destination.stat().st_size != size:
                return False
            if self.verify_checksum:
                return self._sha256(source) == self._sha256(destination)
            return True
        except OSError:
            return False

    def _oldest_pending(self) -> Optional[Path]:
        files = self._files(self.pending_dir)
        if not files:
            return None
        return min(files, key=lambda p: (p.stat().st_mtime, str(p)))

    def _refresh_counters(self) -> None:
        files = self._files(self.pending_dir) + self._files(self.transferring_dir)
        total = 0
        count = 0
        for path in files:
            try:
                total += path.stat().st_size
                count += 1
            except OSError:
                pass
        with self._count_lock:
            self._buffer_bytes = total
            self._pending_count = count

    @staticmethod
    def _files(root: Path) -> list[Path]:
        return [p for p in root.rglob("*") if p.is_file() and not p.name.endswith(".tmp")]

    @staticmethod
    def _safe_relative(path: str) -> Path:
        rel = Path(path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe relative buffer path: {path!r}")
        return rel

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _rate_log(self, event: str, detail: str, interval: float = 60.0) -> None:
        now = time.monotonic()
        if now - self._last_log.get(event, -interval) >= interval:
            self.log(f"{event} {detail}")
            self._last_log[event] = now
