"""Tests for the offline snapshot build/activate hand-off (snapshot_store.py).

Run with: py -m unittest -v
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile

from snapshot_store import (
    CURRENT_POINTER_NAME,
    build_and_activate_snapshot,
    build_snapshot,
    activate_snapshot,
    read_current_snapshot,
    snapshot_index_path,
)


class SnapshotStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.snapshots_dir = Path(self.temporary_directory.name) / "snapshots"
        self.archive_path = Path(self.temporary_directory.name) / "corpus.zip"
        self._write_archive(["First sentence.\n", "Second sentence.\n"])

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_archive(self, lines: list[str]) -> None:
        with zipfile.ZipFile(self.archive_path, "w") as archive:
            archive.writestr("corpus.txt", "".join(lines))

    def test_build_snapshot_writes_a_versioned_directory_without_activating_it(self) -> None:
        result = build_snapshot(self.archive_path, self.snapshots_dir)

        self.assertTrue(result.index_path.is_file())
        self.assertEqual(result.snapshot_directory, self.snapshots_dir / result.version_id)
        self.assertEqual(result.indexed_sentence_count, 2)
        self.assertEqual(result.file_count, 1)
        # Building must not, by itself, make the snapshot the active one: the
        # offline build and the online activation are separate steps.
        self.assertIsNone(read_current_snapshot(self.snapshots_dir))

    def test_activate_snapshot_makes_it_current(self) -> None:
        result = build_snapshot(self.archive_path, self.snapshots_dir)

        activate_snapshot(self.snapshots_dir, result.version_id)

        self.assertEqual(read_current_snapshot(self.snapshots_dir), result.version_id)

    def test_read_current_snapshot_returns_none_when_never_activated(self) -> None:
        self.assertIsNone(read_current_snapshot(self.snapshots_dir))

    def test_activate_snapshot_leaves_no_stray_temporary_pointer_files(self) -> None:
        result = build_snapshot(self.archive_path, self.snapshots_dir)
        activate_snapshot(self.snapshots_dir, result.version_id)

        entries = sorted(path.name for path in self.snapshots_dir.iterdir())
        self.assertEqual(entries, sorted([result.version_id, CURRENT_POINTER_NAME]))

    def test_activate_snapshot_pointer_swap_is_a_single_atomic_rename(self) -> None:
        """The pointer must be written via write-then-rename, not edited in place.

        This is what lets a concurrent reader never observe a half-written
        pointer: it always sees either the previous complete value or the new
        complete value, never a partial one.
        """
        import os

        original_replace = os.replace
        calls: list[tuple[str, str]] = []

        def spy_replace(source: object, destination: object) -> None:
            calls.append((str(source), str(destination)))
            original_replace(source, destination)

        result = build_snapshot(self.archive_path, self.snapshots_dir)
        os.replace = spy_replace
        try:
            activate_snapshot(self.snapshots_dir, result.version_id)
        finally:
            os.replace = original_replace

        self.assertEqual(len(calls), 1)
        source, destination = calls[0]
        self.assertNotEqual(Path(source).name, CURRENT_POINTER_NAME)
        self.assertEqual(Path(destination).name, CURRENT_POINTER_NAME)

    def test_second_snapshot_replaces_the_first_as_current_but_keeps_both_directories(self) -> None:
        first = build_snapshot(self.archive_path, self.snapshots_dir)
        activate_snapshot(self.snapshots_dir, first.version_id)

        self._write_archive(["First sentence.\n", "Second sentence.\n", "Third, added sentence.\n"])
        second = build_snapshot(self.archive_path, self.snapshots_dir)
        activate_snapshot(self.snapshots_dir, second.version_id)

        self.assertEqual(read_current_snapshot(self.snapshots_dir), second.version_id)
        self.assertNotEqual(first.version_id, second.version_id)
        # The old snapshot is left on disk: a reader that only just read the
        # old pointer must still be able to load it.
        self.assertTrue(first.index_path.is_file())
        self.assertTrue(second.index_path.is_file())

    def test_build_snapshot_raises_a_clear_error_for_a_missing_archive(self) -> None:
        with self.assertRaises(FileNotFoundError):
            build_snapshot(self.snapshots_dir / "no-such-archive.zip", self.snapshots_dir)

    def test_build_and_activate_snapshot_does_not_move_the_pointer_when_the_build_fails(self) -> None:
        first = build_snapshot(self.archive_path, self.snapshots_dir)
        activate_snapshot(self.snapshots_dir, first.version_id)

        missing_archive = Path(self.temporary_directory.name) / "missing.zip"
        with self.assertRaises(FileNotFoundError):
            build_and_activate_snapshot(missing_archive, self.snapshots_dir)

        # A failed offline build must never take down a service that is
        # already live on the previous, good snapshot.
        self.assertEqual(read_current_snapshot(self.snapshots_dir), first.version_id)

    def test_build_and_activate_snapshot_flips_the_pointer_on_success(self) -> None:
        result = build_and_activate_snapshot(self.archive_path, self.snapshots_dir)

        self.assertEqual(read_current_snapshot(self.snapshots_dir), result.version_id)

    def test_snapshot_index_path_matches_where_build_snapshot_wrote_the_index(self) -> None:
        result = build_snapshot(self.archive_path, self.snapshots_dir)

        self.assertEqual(
            snapshot_index_path(self.snapshots_dir, result.version_id), result.index_path
        )


if __name__ == "__main__":
    unittest.main()
