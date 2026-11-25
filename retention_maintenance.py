#!/usr/bin/env python3
"""Run local retention independently from the JetsonDualCam GUI."""

import os

from local_retention import LocalRetentionStore


HOME = os.path.expanduser("~")
ROOT = os.environ.get(
    "LOCAL_STORAGE_ROOT", os.path.join(HOME, "Local_Backup", "PeaPLED_SideCAM")
)
MAX_GB = float(os.environ.get("LOCAL_STORAGE_MAX_GB", "60"))
DELETE_GB = float(os.environ.get("LOCAL_STORAGE_DELETE_GB", "30"))


def main() -> int:
    store = LocalRetentionStore(
        ROOT, max_gb=MAX_GB, delete_gb=DELETE_GB,
        delete_pause_seconds=0.01,
    )
    store.run_midnight_maintenance()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
