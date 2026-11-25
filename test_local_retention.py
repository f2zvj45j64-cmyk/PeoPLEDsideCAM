import os
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

from local_retention import LocalRetentionStore


class Clock:
    def __init__(self, value):
        self.value = value

    def now(self):
        return self.value


class LocalRetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.clock = Clock(datetime(2026, 9, 17, 10, 30, 0))
        self.logs = []
        self.store = LocalRetentionStore(
            str(self.root), max_gb=1, delete_gb=0.5,
            log=self.logs.append, now_fn=self.clock.now,
            poll_seconds=0.05, delete_pause_seconds=0,
        )
        self.store.max_bytes = 10
        self.store.delete_bytes = 5

    def tearDown(self):
        self.store.stop()
        self.tmp.cleanup()

    def test_atomic_local_write_and_structure(self):
        path = self.store.create_png("Year_2026/Sep_09/Day_17/LH_CAM/a.png", b"png")
        self.assertEqual(path.read_bytes(), b"png")
        self.assertFalse(any(self.root.rglob("*.tmp")))

    def test_limit_schedules_next_midnight_without_early_delete(self):
        self.store.create_png("old.png", b"123456")
        self.store.create_png("new.png", b"123456")
        status = self.store.status()
        self.assertEqual(status["scheduled_for"], "2026-09-18T00:00:00")
        self.assertFalse(self.store.run_due_job())
        self.assertTrue((self.root / "old.png").exists())
        self.assertTrue((self.root / "new.png").exists())

    def test_due_job_deletes_fifo_until_at_least_target_freed(self):
        old = self.store.create_png("old.png", b"1111")
        middle = self.store.create_png("middle.png", b"2222")
        newest = self.store.create_png("new.png", b"3333")
        base = time.time()
        os.utime(old, (base - 30, base - 30))
        os.utime(middle, (base - 20, base - 20))
        os.utime(newest, (base - 10, base - 10))
        self.clock.value = datetime(2026, 9, 18, 0, 0, 0)
        self.assertTrue(self.store.run_due_job())
        self.assertFalse(old.exists())
        self.assertFalse(middle.exists())
        self.assertTrue(newest.exists())
        self.assertLessEqual(self.store.status()["bytes"], 7)

    def test_schedule_survives_restart(self):
        self.store.create_png("a.png", b"123456")
        self.store.create_png("b.png", b"123456")
        restarted = LocalRetentionStore(
            str(self.root), max_gb=1, delete_gb=0.5,
            log=self.logs.append, now_fn=self.clock.now,
        )
        restarted.max_bytes = 10
        restarted.delete_bytes = 5
        self.assertEqual(restarted.status()["scheduled_for"], "2026-09-18T00:00:00")
        restarted.stop()

    def test_independent_midnight_job_works_without_gui_state(self):
        old = self.store.create_png("old.png", b"111111")
        new = self.store.create_png("new.png", b"222222")
        self.store.state_path.unlink(missing_ok=True)
        self.store._state = {}
        base = time.time()
        os.utime(old, (base - 20, base - 20))
        os.utime(new, (base - 10, base - 10))
        self.assertTrue(self.store.run_midnight_maintenance())
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())

    def test_retention_thread_does_not_block_new_writes(self):
        self.store.delete_pause_seconds = 0.2
        self.store.create_png("old1.png", b"1111")
        self.store.create_png("old2.png", b"2222")
        self.store.create_png("old3.png", b"3333")
        self.clock.value = datetime(2026, 9, 18, 0, 0, 0)
        self.store.start()
        time.sleep(0.03)
        started = time.perf_counter()
        path = self.store.create_png("live.png", b"live")
        elapsed = time.perf_counter() - started
        self.assertTrue(path.exists())
        self.assertLess(elapsed, 0.15)


if __name__ == "__main__":
    unittest.main()
