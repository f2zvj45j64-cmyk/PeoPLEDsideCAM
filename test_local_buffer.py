import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from local_buffer import LocalBuffer


class LocalBufferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.local = base / "local"
        self.network = base / "network"
        self.network.mkdir()
        self.logs = []
        self.buf = LocalBuffer(
            str(self.local), str(self.network), str(self.network),
            require_network_mount=False, log=self.logs.append,
            poll_seconds=0.05, status_seconds=60,
        )

    def tearDown(self):
        self.buf.stop()
        self.tmp.cleanup()

    def test_available_transfer_verify_and_delete(self):
        self.buf.create_png("Year_2026/Sep_09/a.png", b"valid-png-bytes")
        self.assertTrue(self.buf.process_one())
        self.assertEqual((self.network / "Year_2026/Sep_09/a.png").read_bytes(), b"valid-png-bytes")
        self.assertFalse((self.local / "PENDING/Year_2026/Sep_09/a.png").exists())
        self.assertTrue(any(x.startswith("TRANSFER_CONFIRMED") for x in self.logs))
        self.assertTrue(any(x.startswith("LOCAL_COPY_REMOVED") for x in self.logs))

    def test_network_unavailable_keeps_pending(self):
        self.buf.require_network_mount = True
        pending = self.buf.create_png("a.png", b"data")
        self.assertFalse(self.buf.process_one())
        self.assertTrue(pending.exists())
        self.assertFalse((self.network / "a.png").exists())

    def test_network_disappears_during_copy_requeues(self):
        pending = self.buf.create_png("a.png", b"data")
        with mock.patch("local_buffer.shutil.copyfileobj", side_effect=OSError("network lost")):
            self.assertFalse(self.buf.process_one())
        self.assertTrue(pending.exists())
        self.assertFalse((self.network / "a.png").exists())

    def test_network_return_after_outage(self):
        self.buf.require_network_mount = True
        pending = self.buf.create_png("a.png", b"data")
        self.assertFalse(self.buf.process_one())
        self.buf.require_network_mount = False
        self.assertTrue(self.buf.process_one())
        self.assertFalse(pending.exists())
        self.assertEqual((self.network / "a.png").read_bytes(), b"data")

    def test_restart_recovers_transferring(self):
        transferring = self.local / "TRANSFERRING/x/a.png"
        transferring.parent.mkdir(parents=True)
        transferring.write_bytes(b"data")
        self.buf.recover()
        self.assertTrue((self.local / "PENDING/x/a.png").exists())
        self.assertFalse(transferring.exists())

    def test_existing_matching_destination_is_confirmed(self):
        pending = self.buf.create_png("a.png", b"same")
        (self.network / "a.png").write_bytes(b"same")
        self.assertTrue(self.buf.process_one())
        self.assertFalse(pending.exists())

    def test_verification_failure_never_deletes_local(self):
        pending = self.buf.create_png("a.png", b"local-long")
        (self.network / "a.png").write_bytes(b"bad")
        self.assertFalse(self.buf.process_one())
        self.assertTrue(pending.exists())
        self.assertEqual((self.network / "a.png").read_bytes(), b"bad")

    def test_fifo_oldest_first(self):
        older = self.buf.create_png("old.png", b"old")
        newer = self.buf.create_png("new.png", b"new")
        now = time.time()
        os.utime(older, (now - 100, now - 100))
        os.utime(newer, (now, now))
        self.assertTrue(self.buf.process_one())
        self.assertTrue((self.network / "old.png").exists())
        self.assertFalse((self.network / "new.png").exists())

    def test_limit_warnings_do_not_delete(self):
        self.buf.max_bytes = 1
        self.buf.min_free_bytes = 10**30
        pending = self.buf.create_png("a.png", b"data")
        self.assertTrue(pending.exists())
        self.assertTrue(any(x.startswith("BUFFER_LIMIT_WARNING") for x in self.logs))
        self.assertTrue(any(x.startswith("LOW_DISK_SPACE") for x in self.logs))

    def test_partial_tmp_is_not_transferred(self):
        partial = self.local / "PENDING/.a.png.tmp"
        partial.write_bytes(b"partial")
        self.assertFalse(self.buf.process_one())
        self.assertTrue(partial.exists())

    def test_slow_network_worker_does_not_block_new_buffer_write(self):
        import local_buffer
        original_copy = local_buffer.shutil.copyfileobj
        transfer_started = __import__("threading").Event()

        def slow_copy(src, dst, length):
            transfer_started.set()
            time.sleep(0.25)
            return original_copy(src, dst, length)

        with mock.patch("local_buffer.shutil.copyfileobj", side_effect=slow_copy):
            self.buf.start()
            self.buf.create_png("first.png", b"first")
            self.assertTrue(transfer_started.wait(1.0))
            started = time.perf_counter()
            second = self.buf.create_png("second.png", b"second")
            elapsed = time.perf_counter() - started
            self.assertTrue(second.exists())
            self.assertLess(elapsed, 0.15)


if __name__ == "__main__":
    unittest.main()
