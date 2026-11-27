"""Persistent per-project examined/reviewed bout ledger for `jabs.bale`.

Stores user-reviewed bout flags in `<project>/jabs/session/bale_reviewed.json`
using atomic file replacement (`tempfile` + `os.replace`) so examined bouts remain
marked across recalculates, behavior switches, and JABS restarts.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import BoutRecord

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_FILENAME = "bale_reviewed.json"
LEGACY_FILENAME = "label_explorer_reviewed.json"


def _normalize_video(video: str) -> str:
    return Path(str(video)).name


def _canonical_behavior(rec: BoutRecord) -> str:
    orig = str(getattr(rec, "origin_behavior", "") or "").strip()
    if orig:
        return orig
    return str(rec.behavior or "").strip()


def _canonical_origin_source(rec: BoutRecord) -> str:
    orig = str(getattr(rec, "origin_source", "") or "").strip()
    if orig in ("labeled", "predicted"):
        return orig
    return "predicted" if rec.source == "predicted" else "labeled"


def make_bout_signature_key(rec: BoutRecord) -> str:
    """Return a stable string key for `(video, identity, behavior, origin_source, start, end)`."""
    vid = _normalize_video(rec.video)
    beh = _canonical_behavior(rec)
    src = _canonical_origin_source(rec)
    return f"{vid}|{int(rec.identity)}|{beh}|{src}|{int(rec.start_frame)}|{int(rec.end_frame)}"


def _frame_overlap_matches(s1: int, e1: int, s2: int, e2: int) -> bool:
    """Return True if two inclusive frame spans overlap significantly (`IoU >= 0.50` or `>= 70%` of shorter)."""
    ov = min(e1, e2) - max(s1, s2) + 1
    if ov <= 0:
        return False
    len1 = max(1, e1 - s1 + 1)
    len2 = max(1, e2 - s2 + 1)
    union = len1 + len2 - ov
    if (ov / max(1, union)) >= 0.50:
        return True
    return (ov / min(len1, len2)) >= 0.70


class ExaminedBoutStore:
    """Project-scoped persistent ledger of bouts examined or triaged in BALE (Bout & Label Explorer)."""

    def __init__(self, project_dir: Path | str) -> None:
        self._project_dir = Path(project_dir)
        self._path = self._project_dir / "jabs" / "session" / DEFAULT_FILENAME
        self._entries: dict[str, dict[str, Any]] = {}
        self._by_track_bucket: dict[tuple[str, int, str, str], dict[str, None]] = {}
        self._key_to_bucket: dict[str, tuple[str, int, str, str]] = {}
        self.load()

    @property
    def path(self) -> Path:
        """Return the path to the persistent examined bout ledger JSON file."""
        return self._path

    @staticmethod
    def _entry_bucket(ent: dict[str, Any]) -> tuple[str, int, str, str]:
        return (
            _normalize_video(str(ent.get("video", ""))),
            int(ent.get("identity", 0)),
            str(ent.get("behavior", "")),
            str(ent.get("origin_source", "labeled")),
        )

    def _index_add(self, key: str, ent: dict[str, Any]) -> None:
        old_bucket = self._key_to_bucket.pop(key, None)
        if old_bucket is not None:
            old_map = self._by_track_bucket.get(old_bucket)
            if old_map is not None:
                old_map.pop(key, None)
                if not old_map:
                    self._by_track_bucket.pop(old_bucket, None)
        bucket = self._entry_bucket(ent)
        self._by_track_bucket.setdefault(bucket, {})[key] = None
        self._key_to_bucket[key] = bucket

    def _index_remove(self, key: str) -> None:
        bucket = self._key_to_bucket.pop(key, None)
        if bucket is None:
            return
        bucket_map = self._by_track_bucket.get(bucket)
        if bucket_map is not None:
            bucket_map.pop(key, None)
            if not bucket_map:
                self._by_track_bucket.pop(bucket, None)

    def _rebuild_index(self) -> None:
        buckets: dict[tuple[str, int, str, str], dict[str, None]] = {}
        key_to_bucket: dict[str, tuple[str, int, str, str]] = {}
        for key, ent in self._entries.items():
            if not isinstance(ent, dict):
                continue
            bucket = self._entry_bucket(ent)
            buckets.setdefault(bucket, {})[key] = None
            key_to_bucket[key] = bucket
        self._by_track_bucket = buckets
        self._key_to_bucket = key_to_bucket

    def load(self) -> None:
        """Load persisted examined flags from disk (with legacy filename fallback)."""
        self._entries = {}
        source_path = self._path
        if not source_path.is_file():
            legacy_path = self._path.parent / LEGACY_FILENAME
            if legacy_path.is_file():
                source_path = legacy_path
        if source_path.is_file():
            try:
                raw = json.loads(source_path.read_text(encoding="utf-8"))
                entries = raw.get("entries", {}) if isinstance(raw, dict) else {}
                if isinstance(entries, dict):
                    self._entries = {
                        str(k): dict(v) for k, v in entries.items() if isinstance(v, dict)
                    }
            except Exception as exc:
                logger.warning("Could not read examined bout ledger %s: %s", source_path, exc)
                self._entries = {}
        self._rebuild_index()

    def save(self) -> bool:
        """Persist examined flags atomically to `<project>/jabs/session/bale_reviewed.json`."""
        tmp_path: Path | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "entries": self._entries,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            fd, tmp_name = tempfile.mkstemp(
                prefix=".bale_reviewed.",
                suffix=".tmp",
                dir=str(self._path.parent),
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
            os.replace(tmp_path, self._path)
            return True
        except OSError:
            logger.warning(
                "Could not save examined bout ledger %s",
                self._path,
                exc_info=True,
            )
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    tmp_path.unlink(missing_ok=True)
            return False

    def find_match_key(self, rec: BoutRecord) -> str | None:
        """Find exact key match, or overlap match for shifted predicted bouts."""
        exact = make_bout_signature_key(rec)
        if exact in self._entries:
            return exact

        src = _canonical_origin_source(rec)
        if src != "predicted":
            return None

        bucket = (_normalize_video(rec.video), int(rec.identity), _canonical_behavior(rec), src)
        s1, e1 = int(rec.start_frame), int(rec.end_frame)
        for cand_key in self._by_track_bucket.get(bucket, ()):
            ent = self._entries.get(cand_key)
            if ent is None:
                continue
            s2 = int(ent.get("start_frame", -1))
            e2 = int(ent.get("end_frame", -1))
            if _frame_overlap_matches(s1, e1, s2, e2):
                return cand_key
        return None

    def find_match(self, rec: BoutRecord) -> dict[str, Any] | None:
        """Return a copy of the persisted examined entry matching `rec`, or `None`."""
        key = self.find_match_key(rec)
        if key is None:
            return None
        return dict(self._entries[key])

    def mark(
        self,
        rec: BoutRecord,
        *,
        action: str = "examined",
        note: str = "✓ Examined",
        persist: bool = True,
    ) -> dict[str, Any]:
        """Record `rec` as examined/reviewed and persist to disk."""
        key = make_bout_signature_key(rec)
        entry = {
            "key": key,
            "video": _normalize_video(rec.video),
            "identity": int(rec.identity),
            "behavior": _canonical_behavior(rec),
            "origin_source": _canonical_origin_source(rec),
            "start_frame": int(rec.start_frame),
            "end_frame": int(rec.end_frame),
            "action": str(action),
            "note": str(note),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._entries[key] = entry
        self._index_add(key, entry)
        if persist:
            self.save()
        return dict(entry)

    def mark_many(
        self,
        records: list[BoutRecord],
        *,
        action: str = "examined",
        note: str = "✓ Examined",
    ) -> int:
        """Batch-mark multiple bouts with a single atomic disk write."""
        if not records:
            return 0
        now_iso = datetime.now().isoformat(timespec="seconds")
        for rec in records:
            key = make_bout_signature_key(rec)
            entry = {
                "key": key,
                "video": _normalize_video(rec.video),
                "identity": int(rec.identity),
                "behavior": _canonical_behavior(rec),
                "origin_source": _canonical_origin_source(rec),
                "start_frame": int(rec.start_frame),
                "end_frame": int(rec.end_frame),
                "action": str(action),
                "note": str(note),
                "updated_at": now_iso,
            }
            self._entries[key] = entry
        self._rebuild_index()
        self.save()
        return len(records)

    def unmark(self, rec: BoutRecord, *, persist: bool = True) -> dict[str, Any] | None:
        """Remove examined flag for `rec` (exact or overlapping predicted match) and persist."""
        key = self.find_match_key(rec)
        if key is None:
            return None
        removed = self._entries.pop(key, None)
        self._index_remove(key)
        if persist:
            self.save()
        return dict(removed) if removed is not None else None

    def restore_entry(self, entry: dict[str, Any] | None, *, persist: bool = True) -> None:
        """Restore a previously snapshotted entry (used by Undo)."""
        if not entry or not entry.get("key"):
            return
        key = str(entry["key"])
        ent_copy = dict(entry)
        self._entries[key] = ent_copy
        self._index_add(key, ent_copy)
        if persist:
            self.save()

    def clear(self, *, behavior: str | None = None) -> int:
        """Clear examined flags for a single behavior or the entire project."""
        if behavior is None:
            n = len(self._entries)
            if n == 0:
                return 0
            self._entries = {}
            self._rebuild_index()
            self.save()
            return n

        target = str(behavior).strip()
        keys_to_drop = [
            k for k, ent in self._entries.items() if str(ent.get("behavior", "")).strip() == target
        ]
        if not keys_to_drop:
            return 0
        for k in keys_to_drop:
            self._entries.pop(k, None)
        self._rebuild_index()
        self.save()
        return len(keys_to_drop)

    def count(self, *, behavior: str | None = None) -> int:
        """Return the number of persisted examined entries (optionally filtered by behavior)."""
        if behavior is None:
            return len(self._entries)
        target = str(behavior).strip()
        return sum(
            1 for ent in self._entries.values() if str(ent.get("behavior", "")).strip() == target
        )
