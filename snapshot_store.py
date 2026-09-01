"""Filesystem hand-off between the offline indexer and the online server.

The offline phase (see ``build_snapshot.py``) writes each new or updated
corpus into its own versioned directory under a shared ``snapshots_dir``
instead of overwriting whatever the running service is currently reading.
Once a build is complete and validated, ``activate_snapshot`` atomically
flips a small pointer file, ``CURRENT``, to name that version.

The online service (see ``web_app.py``'s ``SnapshotWatcher``) only ever reads
``CURRENT`` and loads whatever version it names. Because the pointer is
written with write-then-atomic-rename, a reader never observes a half
written value, and because old snapshot directories are left in place, a
reader that read the pointer just before a new version was activated can
still finish loading the previous, complete snapshot. This is what allows a
new data source to be added live and remotely with no service downtime: the
offline build never touches the directory the online service is reading.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import time
from typing import Optional
import uuid

from autocomplete import AutocompleteEngine


CURRENT_POINTER_NAME = "CURRENT"
INDEX_FILE_NAME = "index.sqlite3"


@dataclass(frozen=True)
class SnapshotBuildResult:
    """What the offline phase produced for one versioned snapshot."""

    version_id: str
    snapshot_directory: Path
    index_path: Path
    indexed_sentence_count: int
    file_count: int


def _compute_version_id(archive_path: Path) -> str:
    """A sortable, unique snapshot name: build timestamp plus a content hash.

    The hash ties a version id to the exact bytes it was built from; the
    timestamp keeps directory listings in build order even when a corpus is
    rebuilt unchanged.
    """

    digest = hashlib.sha256()
    with archive_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return "{0}-{1}".format(timestamp, digest.hexdigest()[:12])


def snapshot_index_path(snapshots_dir: Path, version_id: str) -> Path:
    """Where ``build_snapshot`` writes (and the watcher reads) one version's index."""

    return Path(snapshots_dir) / version_id / INDEX_FILE_NAME


def build_snapshot(archive_path: Path, snapshots_dir: Path) -> SnapshotBuildResult:
    """Offline phase: build a new versioned snapshot. Does not activate it.

    Building into a brand-new directory, rather than into the currently
    active one, is what lets this run at any time -- including while the
    online service is live on an older snapshot -- without disturbing it.
    """

    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError("Archive was not found: {0}".format(archive_path))

    snapshots_dir = Path(snapshots_dir)
    version_id = _compute_version_id(archive_path)
    snapshot_directory = snapshots_dir / version_id
    snapshot_directory.mkdir(parents=True)
    index_path = snapshot_directory / INDEX_FILE_NAME

    engine = AutocompleteEngine.from_archive(archive_path, index_path=index_path, rebuild_index=True)
    try:
        if engine.indexed_sentence_count == 0:
            raise ValueError(
                "The archive produced an empty index; refusing to publish an empty snapshot."
            )
        return SnapshotBuildResult(
            version_id=version_id,
            snapshot_directory=snapshot_directory,
            index_path=index_path,
            indexed_sentence_count=engine.indexed_sentence_count,
            file_count=engine.file_count,
        )
    finally:
        engine.close()


def activate_snapshot(snapshots_dir: Path, version_id: str) -> None:
    """Atomically make ``version_id`` the snapshot the online service loads.

    The pointer is never edited in place: it is written to a fresh temporary
    file in the same directory (so the rename below stays on one filesystem)
    and then moved onto ``CURRENT`` with ``os.replace``, which is an atomic
    rename on both POSIX and Windows. A concurrent reader therefore always
    sees either the previous complete pointer value or the new one.
    """

    snapshots_dir = Path(snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = snapshots_dir / CURRENT_POINTER_NAME
    temporary_path = snapshots_dir / ".{0}.{1}.tmp".format(CURRENT_POINTER_NAME, uuid.uuid4().hex)
    temporary_path.write_text(version_id, encoding="utf-8")
    os.replace(temporary_path, pointer_path)


def read_current_snapshot(snapshots_dir: Path) -> Optional[str]:
    """Return the currently active version id, or ``None`` if none was ever activated."""

    pointer_path = Path(snapshots_dir) / CURRENT_POINTER_NAME
    try:
        content = pointer_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return content or None


def build_and_activate_snapshot(archive_path: Path, snapshots_dir: Path) -> SnapshotBuildResult:
    """The full offline stage: build a snapshot, then activate it once it validates.

    If the build raises, ``CURRENT`` is left untouched, so a service already
    live on a previous snapshot is unaffected by a failed or in-progress
    build of a new one.
    """

    result = build_snapshot(archive_path, snapshots_dir)
    activate_snapshot(snapshots_dir, result.version_id)
    return result
