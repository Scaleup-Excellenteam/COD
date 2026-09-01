"""Offline phase for zero-downtime indexing: build and activate a snapshot.

Run this whenever a new or updated data source is ready to publish -- for
example, after a new "Archive (N).zip" lands next to this script, whether
copied in locally or dropped in remotely by any existing upload mechanism.
It builds a fresh index into its own versioned directory under
--snapshots-dir and only then atomically flips the CURRENT pointer there.

A web_app.py already running with --snapshots-dir pointed at the same
directory notices the flip and hot-swaps live, with no restart: this is the
whole zero-downtime hand-off, end to end.

    py build_snapshot.py --archive "Archive (2).zip" --snapshots-dir snapshots
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from snapshot_store import build_and_activate_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("Archive (2).zip"),
        help="Path to the ZIP archive containing .txt files for this snapshot.",
    )
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=Path("snapshots"),
        help="Directory holding versioned snapshots and the CURRENT pointer.",
    )
    arguments = parser.parse_args()

    print("Building a new snapshot from {0}...".format(arguments.archive), flush=True)
    started_at = time.perf_counter()
    result = build_and_activate_snapshot(arguments.archive, arguments.snapshots_dir)
    print(
        "Snapshot {0} activated: {1:,} lines from {2:,} files in {3:.1f}s.".format(
            result.version_id,
            result.indexed_sentence_count,
            result.file_count,
            time.perf_counter() - started_at,
        )
    )
    print(
        "Any web_app.py running with --snapshots-dir {0} will pick this up live.".format(
            arguments.snapshots_dir
        )
    )


if __name__ == "__main__":
    main()
