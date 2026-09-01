"""Tests for the zero-downtime snapshot watcher and engine hot-swap in web_app.py.

These exercise the watcher's decision logic directly, against a lightweight
stand-in for the parts of ``HTTPServer`` it touches, rather than opening a
real socket -- consistent with how ``test_autocomplete.py`` calls ``run_cli``
directly instead of spawning a process.

Run with: py -m unittest -v
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
import zipfile

from autocomplete import AutocompleteEngine
import snapshot_store
from web_app import SnapshotWatcher, _activate_engine, _is_manual_upload_allowed


class FakeServer(SimpleNamespace):
    """Stands in for the subset of HTTPServer state the watcher touches."""


class SnapshotWatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.snapshots_dir = Path(self.temporary_directory.name) / "snapshots"
        self.archive_path = Path(self.temporary_directory.name) / "corpus.zip"
        self._write_archive(["First sentence.\n"])
        self.first_result = snapshot_store.build_and_activate_snapshot(
            self.archive_path, self.snapshots_dir
        )
        self.engine_lock = threading.Lock()
        self.server = FakeServer(
            engine=AutocompleteEngine.open_snapshot(self.first_result.index_path),
            active_index_name=self.first_result.version_id,
            snapshots_dir=self.snapshots_dir,
        )

    def tearDown(self) -> None:
        self.server.engine.close()
        self.temporary_directory.cleanup()

    def _write_archive(self, lines: list[str]) -> None:
        with zipfile.ZipFile(self.archive_path, "w") as archive:
            archive.writestr("corpus.txt", "".join(lines))

    def _watcher(self, server: FakeServer, snapshots_dir: Path, poll_interval_seconds: float = 0.01) -> SnapshotWatcher:
        return SnapshotWatcher(server, snapshots_dir, self.engine_lock, poll_interval_seconds)

    def test_poll_once_does_nothing_when_the_pointer_is_unchanged(self) -> None:
        watcher = self._watcher(self.server, self.snapshots_dir)

        self.assertFalse(watcher.poll_once())
        self.assertEqual(self.server.active_index_name, self.first_result.version_id)

    def test_poll_once_hot_swaps_when_a_new_snapshot_is_activated(self) -> None:
        self._write_archive(["First sentence.\n", "Second, added sentence.\n"])
        second_result = snapshot_store.build_and_activate_snapshot(self.archive_path, self.snapshots_dir)
        watcher = self._watcher(self.server, self.snapshots_dir)

        swapped = watcher.poll_once()

        self.assertTrue(swapped)
        self.assertEqual(self.server.active_index_name, second_result.version_id)
        self.assertTrue(self.server.engine.get_best_k_completions("added sentence"))

    def test_poll_once_keeps_serving_the_old_snapshot_when_the_new_one_is_corrupt(self) -> None:
        broken_version_id = "broken-version"
        broken_directory = self.snapshots_dir / broken_version_id
        broken_directory.mkdir()
        (broken_directory / "index.sqlite3").write_bytes(b"not a real sqlite file")
        snapshot_store.activate_snapshot(self.snapshots_dir, broken_version_id)
        watcher = self._watcher(self.server, self.snapshots_dir)

        swapped = watcher.poll_once()

        # A bad pointer must never take a live service down: it keeps
        # serving the last good snapshot and simply tries again next poll.
        self.assertFalse(swapped)
        self.assertEqual(self.server.active_index_name, self.first_result.version_id)
        self.assertTrue(self.server.engine.get_best_k_completions("first sentence"))

    def test_poll_once_does_nothing_when_no_snapshot_was_ever_activated(self) -> None:
        empty_snapshots_dir = Path(self.temporary_directory.name) / "empty-snapshots"
        server = FakeServer(engine=self.server.engine, active_index_name=None, snapshots_dir=empty_snapshots_dir)
        watcher = self._watcher(server, empty_snapshots_dir)

        self.assertFalse(watcher.poll_once())

    def test_start_and_stop_run_the_watcher_on_a_background_thread(self) -> None:
        self._write_archive(["First sentence.\n", "Second, added sentence.\n"])
        second_result = snapshot_store.build_and_activate_snapshot(self.archive_path, self.snapshots_dir)
        watcher = self._watcher(self.server, self.snapshots_dir, poll_interval_seconds=0.02)

        watcher.start()
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and self.server.active_index_name != second_result.version_id:
                time.sleep(0.01)
        finally:
            watcher.stop()

        self.assertEqual(self.server.active_index_name, second_result.version_id)


class ActivateEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.archive_path = Path(self.temporary_directory.name) / "corpus.zip"
        with zipfile.ZipFile(self.archive_path, "w") as archive:
            archive.writestr("corpus.txt", "One sentence.\n")
        self.old_engine = AutocompleteEngine.from_archive(self.archive_path)
        self.new_engine = AutocompleteEngine.from_archive(self.archive_path)
        self.engine_lock = threading.Lock()
        self.server = FakeServer(engine=self.old_engine, active_index_name="old")

    def tearDown(self) -> None:
        self.server.engine.close()
        self.temporary_directory.cleanup()

    def test_activate_engine_swaps_the_reference_and_closes_the_old_engine(self) -> None:
        _activate_engine(self.server, self.new_engine, "new", self.engine_lock)

        self.assertIs(self.server.engine, self.new_engine)
        self.assertEqual(self.server.active_index_name, "new")
        # The old engine is closed only *after* the swap, never before or
        # concurrently with a request that might still be reading from it.
        with self.assertRaises(RuntimeError):
            self.old_engine.get_best_k_completions("one")

    def test_activate_engine_waits_for_an_in_flight_request_to_release_the_lock(self) -> None:
        """A request holding the lock must finish before the swap becomes visible.

        This is the guarantee that makes the hot-swap safe: the watcher
        thread cannot swap the engine reference (and later close the old
        connection) while the request-handling thread is still inside a
        query that started against the old engine.
        """

        swap_thread = threading.Thread(
            target=_activate_engine, args=(self.server, self.new_engine, "new", self.engine_lock)
        )
        with self.engine_lock:
            swap_thread.start()
            time.sleep(0.05)
            # The lock is still held here, so the swap must not have happened.
            self.assertIs(self.server.engine, self.old_engine)
            self.assertEqual(self.server.active_index_name, "old")
        swap_thread.join(timeout=1.0)

        self.assertIs(self.server.engine, self.new_engine)


class ManualUploadModeTests(unittest.TestCase):
    def test_manual_upload_is_allowed_without_a_snapshots_dir(self) -> None:
        server = FakeServer(snapshots_dir=None)
        self.assertTrue(_is_manual_upload_allowed(server))

    def test_manual_upload_is_disabled_in_snapshot_mode(self) -> None:
        server = FakeServer(snapshots_dir=Path("snapshots"))
        self.assertFalse(_is_manual_upload_allowed(server))


if __name__ == "__main__":
    unittest.main()
