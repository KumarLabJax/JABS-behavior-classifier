r"""Flag likely instancing (segmentation and tracking) errors in pose files, for training data.

Scans a directory of v8 pose files (bounding boxes are required) for frames where the
segmentation or identity tracking probably failed, and saves those frames so they can
be added to pose/segmentation training data. Four kinds of event are found:

``fragment``
    An identity's track ends for good and another identity's track starts within a
    few frames in the same place, e.g. after a segmentation failure splits one animal
    into two and the animal continues under a new identity.

``flicker``
    Identities repeatedly dropping out and reappearing in the same place, typically
    animals huddled together. These are common, so frames are extracted for at most
    one episode per video (the one with the most dropouts) and at most
    ``--max-flicker`` overall.

``extra``
    Frames with more detected animals than the video's expected count (the number of
    identities present in at least 10% of its frames), e.g. one animal split into two
    masks, or a duplicate detection. These are common too, so frames are extracted for
    at most ``--max-extra-per-video`` per video (the longest) and ``--max-extra``
    overall.

``short``
    An identity present for only a few frames in the whole video.

Events over a cap are still listed in ``events.csv``, without frames. For each
extracted event's key frames, the output directory receives:

* ``<video>_<frame>.png``: the raw frame
* ``<video>_<frame>_overlay.png``: the frame with pose, segmentation, and each
  identity's bounding box labeled with its identity number

plus ``events.csv`` with one row per event. Videos are not copied; ``events.csv``
names the video for each event. Pose files are copied to a temporary directory before
they are opened, so the inputs are never modified.

Usage::

    python dev/flag_instancing_errors.py POSE_DIR VIDEO_DIR OUTPUT_DIR

    python dev/flag_instancing_errors.py /data/new-pose /data/project /data/instancing_errors \\
        --max-flicker 30 --max-extra 100 --max-extra-per-video 3 --workers 8
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import tempfile
import warnings
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import click
import cv2
import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.table import Table

from jabs.pose_estimation import PoseEstimation, open_pose_file
from jabs.video_export.frame_renderer import render_overlay_frame

logger = logging.getLogger(__name__)

POSE_SUFFIX = "_pose_est_v8.h5"
VIDEO_SUFFIXES = (".mp4", ".avi")  # the video types JABS projects support
HANDOFF_GAP = 3  # max frames between one track ending and another starting
EPISODE_GAP = 10  # handoffs closer than this many frames belong to one episode
LASTING = 30  # a fragment's old identity stays gone, and its new one was absent, this long
SHORT_FRAMES = 30  # identities present for fewer frames than this are flagged
EXPECTED_PRESENCE = 0.1  # identities present in this fraction of frames count as animals
BOX_IOU_SAME_PLACE = 0.2

# BGR colors for identity boxes in the overlay frames
_PALETTE = [(0, 0, 255), (0, 200, 0), (255, 64, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]

Box = npt.NDArray[np.floating]  # one box as [[x1, y1], [x2, y2]]


@dataclass
class FailureEvent:
    """One likely failure in one video.

    Attributes:
        video: Video filename.
        pose_file: Pose file the event was found in.
        event: Event type: fragment, flicker, extra, or short.
        start_frame: First frame of the event.
        end_frame: Last frame of the event, inclusive.
        identities: Identities involved, e.g. ``"1->2"`` or ``"0,1"``.
        detail: Human-readable description.
        key_frames: Frames to extract for this event.
        handoffs: Number of dropout/reappearance handoffs (flicker episodes only).
        frames_extracted: Raw frame filenames written for this event.
        note: Why frames were not extracted, if they were not.
    """

    video: str
    pose_file: Path
    event: str
    start_frame: int
    end_frame: int
    identities: str
    detail: str
    key_frames: list[int]
    handoffs: int = 0
    frames_extracted: list[str] = field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class _Handoff:
    """One identity's track ending as another's starts in the same place."""

    frame: int
    from_id: int
    from_end: int
    to_id: int
    to_start: int
    iou: float


def _runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return the inclusive (start, end) frame ranges where ``mask`` is True."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def _box_iou(a: Box, b: Box) -> float:
    """Return the IoU of two boxes."""
    (ax1, ay1), (ax2, ay2) = a
    (bx1, by1), (bx2, by2) = b
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def _center_inside(a: Box, b: Box) -> bool:
    """Return whether the center of box ``a`` lies inside box ``b``."""
    cx, cy = a.mean(axis=0)
    (bx1, by1), (bx2, by2) = b
    return bool(bx1 <= cx <= bx2 and by1 <= cy <= by2)


def _same_place(a: Box, b: Box) -> tuple[bool, float]:
    """Return whether two boxes plausibly cover the same animal, and their IoU."""
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return False, 0.0
    iou = _box_iou(a, b)
    return iou > BOX_IOU_SAME_PLACE or _center_inside(a, b) or _center_inside(b, a), iou


def _find_handoffs(
    present: dict[int, npt.NDArray[np.bool_]], boxes: dict[int, npt.NDArray[np.floating]]
) -> list[_Handoff]:
    """Find every place one identity's track ends as another's starts nearby."""
    n_frames = len(next(iter(present.values())))
    segments = {i: _runs(mask) for i, mask in present.items()}
    handoffs = []
    for a, a_segments in segments.items():
        for _, a_end in a_segments:
            if a_end >= n_frames - 1:
                continue
            for b, b_segments in segments.items():
                if b == a:
                    continue
                for b_start, _ in b_segments:
                    if b_start == 0 or not a_end - HANDOFF_GAP <= b_start <= a_end + HANDOFF_GAP:
                        continue
                    same, iou = _same_place(boxes[a][a_end], boxes[b][b_start])
                    if same:
                        handoffs.append(_Handoff(min(a_end, b_start), a, a_end, b, b_start, iou))
    return sorted(handoffs, key=lambda h: (h.frame, h.from_id, h.to_id))


def _episode_events(
    handoffs: list[_Handoff],
    present: dict[int, npt.NDArray[np.bool_]],
    video: str,
    pose_file: Path,
) -> list[FailureEvent]:
    """Group handoffs into episodes; a lone, lasting handoff is a fragment, else flicker.

    A handoff joins an episode only if it is within ``EPISODE_GAP`` frames of the
    episode's latest handoff and involves one of the episode's identities, so
    unrelated handoffs that happen close together stay separate events.
    """
    episodes: list[list[_Handoff]] = []
    for handoff in handoffs:
        ids = {handoff.from_id, handoff.to_id}
        episode = next(
            (
                e
                for e in reversed(episodes)
                if handoff.frame - e[-1].frame <= EPISODE_GAP
                and ids & ({h.from_id for h in e} | {h.to_id for h in e})
            ),
            None,
        )
        if episode is not None:
            episode.append(handoff)
        else:
            episodes.append([handoff])

    events = []
    for episode in episodes:
        start = episode[0].frame
        end = max(max(h.from_end, h.to_start) for h in episode)
        if len(episode) == 1:
            h = episode[0]
            old_gone = not present[h.from_id][h.from_end + 1 : h.from_end + 1 + LASTING].any()
            new_is_new = not present[h.to_id][max(0, h.to_start - LASTING) : h.to_start].any()
            if old_gone and new_is_new:
                events.append(
                    FailureEvent(
                        video=video,
                        pose_file=pose_file,
                        event="fragment",
                        start_frame=start,
                        end_frame=end,
                        identities=f"{h.from_id}->{h.to_id}",
                        detail=(
                            f"id {h.from_id} ends at {h.from_end}, id {h.to_id} starts at "
                            f"{h.to_start}, box IoU {h.iou:.2f}"
                        ),
                        key_frames=sorted({h.from_end, h.to_start}),
                    )
                )
                continue
        involved = sorted({h.from_id for h in episode} | {h.to_id for h in episode})
        events.append(
            FailureEvent(
                video=video,
                pose_file=pose_file,
                event="flicker",
                start_frame=start,
                end_frame=end,
                identities=",".join(map(str, involved)),
                detail=f"{len(episode)} dropout/reappearance handoffs between ids {involved}",
                key_frames=[(start + end) // 2],
                handoffs=len(episode),
            )
        )
    return events


def _extra_detection_events(
    present: dict[int, npt.NDArray[np.bool_]], video: str, pose_file: Path
) -> list[FailureEvent]:
    """Find runs of frames with more animals than the video's expected count.

    The expected count is the number of identities present in at least
    ``EXPECTED_PRESENCE`` of the frames. The most common per-frame count is not used
    because animals that are merged into one detection for most of a video would make
    every correctly separated frame look like an extra detection.
    """
    count = np.sum(list(present.values()), axis=0)
    if not count.any():
        return []
    usual = max(1, sum(1 for mask in present.values() if mask.mean() >= EXPECTED_PRESENCE))
    return [
        FailureEvent(
            video=video,
            pose_file=pose_file,
            event="extra",
            start_frame=start,
            end_frame=end,
            identities=",".join(str(i) for i, mask in present.items() if mask[start]),
            detail=(
                f"{int(count[start])} animals detected (expected {usual}) "
                f"for {end - start + 1} frames"
            ),
            key_frames=[(start + end) // 2],
        )
        for start, end in _runs(count > usual)
    ]


def _short_identity_events(
    present: dict[int, npt.NDArray[np.bool_]], video: str, pose_file: Path
) -> list[FailureEvent]:
    """Find identities present for fewer than SHORT_FRAMES frames in total."""
    events = []
    for identity, mask in present.items():
        frames = np.flatnonzero(mask)
        if 0 < len(frames) < SHORT_FRAMES:
            events.append(
                FailureEvent(
                    video=video,
                    pose_file=pose_file,
                    event="short",
                    start_frame=int(frames[0]),
                    end_frame=int(frames[-1]),
                    identities=str(identity),
                    detail=f"id {identity} present for only {len(frames)} frames",
                    key_frames=[int(frames[0])],
                )
            )
    return events


def find_events(pose: PoseEstimation, video: str, pose_file: Path) -> list[FailureEvent]:
    """Find likely failure events in one pose file.

    Args:
        pose: Pose estimation for the video (v8, with bounding boxes).
        video: Video filename, recorded in each event.
        pose_file: Original pose file path, recorded in each event.

    Returns:
        All fragment, flicker, extra-detection, and short-identity events.
    """
    present = {i: np.asarray(pose.identity_mask(i), dtype=bool) for i in pose.identities}
    if not present:
        return []
    boxes = {i: pose.get_bounding_boxes(i) for i in pose.identities}
    return (
        _episode_events(_find_handoffs(present, boxes), present, video, pose_file)
        + _extra_detection_events(present, video, pose_file)
        + _short_identity_events(present, video, pose_file)
    )


@contextmanager
def _open_pose_copy(pose_path: Path) -> Iterator[PoseEstimation]:
    """Open a temporary copy of ``pose_path`` so the original is never written."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / pose_path.name
        shutil.copy(pose_path, copy)
        yield open_pose_file(copy)


def _video_name(pose_path: Path, video_dir: Path) -> str:
    """Return the filename of the video for ``pose_path`` in ``video_dir``.

    Falls back to the ``.mp4`` name when no supported video is found, so the event
    still names the video it expected.
    """
    stem = pose_path.name[: -len(POSE_SUFFIX)]
    for suffix in VIDEO_SUFFIXES:
        if (video_dir / f"{stem}{suffix}").exists():
            return f"{stem}{suffix}"
    return f"{stem}{VIDEO_SUFFIXES[0]}"


def _detect(job: tuple[Path, Path]) -> list[FailureEvent]:
    """Find the events in one pose file (worker function; reads no video frames)."""
    warnings.simplefilter("ignore")
    pose_path, video_dir = job
    video = _video_name(pose_path, video_dir)
    try:
        with _open_pose_copy(pose_path) as pose:
            if not getattr(pose, "has_bounding_boxes", False):
                logger.warning("%s has no bounding boxes; skipping", pose_path.name)
                return []
            return find_events(pose, video, pose_path)
    except Exception:
        logger.exception("Failed to scan %s", pose_path)
        return []


def _draw_overlay(image: npt.NDArray[np.uint8], pose: PoseEstimation, frame: int) -> npt.NDArray:
    """Draw pose, segmentation, labeled identity boxes, and the frame number."""
    out = render_overlay_frame(image, pose, frame, draw_pose=True, draw_segmentation=True)
    for identity in pose.identities:
        boxes = pose.get_bounding_boxes(identity)
        if boxes is None or not np.isfinite(boxes[frame]).all():
            continue
        (x1, y1), (x2, y2) = boxes[frame]
        if x2 <= x1 or y2 <= y1:
            continue
        color = _PALETTE[identity % len(_PALETTE)]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            out,
            f"id {identity}",
            (int(x1), max(int(y1) - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    cv2.putText(out, f"frame {frame}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def _extract(job: tuple[Path, Path, list[int], Path]) -> set[int]:
    """Save raw and overlay PNGs of the given frames of one video (worker function).

    Frames are read sequentially rather than by seeking, which is not frame-exact for
    all codecs.
    """
    warnings.simplefilter("ignore")
    pose_path, video, frames, out_dir = job
    wanted, saved = set(frames), set()
    with _open_pose_copy(pose_path) as pose:
        reader = cv2.VideoCapture(str(video))
        if not reader.isOpened():
            logger.error("Unable to open video %s", video)
            return saved
        index = 0
        while wanted:
            ok, image = reader.read()
            if not ok:
                break
            if index in wanted:
                base = out_dir / f"{video.stem}_{index:05d}"
                raw_ok = cv2.imwrite(f"{base}.png", image)
                overlay_ok = cv2.imwrite(f"{base}_overlay.png", _draw_overlay(image, pose, index))
                if raw_ok and overlay_ok:
                    saved.add(index)
                else:
                    logger.error(
                        "Failed to write %s (raw written: %s, overlay written: %s)",
                        base,
                        raw_ok,
                        overlay_ok,
                    )
                wanted.discard(index)
            index += 1
        reader.release()
    if wanted:
        logger.warning("%s: frames past the end of the video: %s", video.name, sorted(wanted))
    return saved


def cap_events(
    events: list[FailureEvent],
    kind: str,
    per_video: int,
    overall: int,
) -> None:
    """Keep frames for at most ``per_video`` events of ``kind`` per video and ``overall`` total.

    Events are ranked by size: handoff count for flicker episodes, run length for
    everything else. The largest events in each video are kept, and the largest of
    those across videos win. Events that are not kept keep their row in
    ``events.csv`` but no frames.

    Args:
        events: All events; events of ``kind`` that are not kept are modified in place.
        kind: Event type to cap.
        per_video: Maximum number of events of ``kind`` to keep per video.
        overall: Maximum number of events of ``kind`` to keep in total.
    """

    def size(event: FailureEvent) -> int:
        return event.handoffs if kind == "flicker" else event.end_frame - event.start_frame + 1

    by_video: dict[str, list[FailureEvent]] = {}
    for event in events:
        if event.event == kind:
            by_video.setdefault(event.video, []).append(event)
    per_video_kept = [
        e
        for video_events in by_video.values()
        for e in sorted(video_events, key=lambda e: (-size(e), e.start_frame))[:per_video]
    ]
    chosen = sorted(per_video_kept, key=lambda e: (-size(e), e.video, e.start_frame))[:overall]
    chosen_ids = {id(e) for e in chosen}
    for event in events:
        if event.event == kind and id(event) not in chosen_ids:
            event.key_frames = []
            event.note = f"frames not extracted ({kind} cap)"


def _write_csv(events: list[FailureEvent], path: Path) -> None:
    """Write one row per event to ``path``."""
    fields = [
        "video",
        "event",
        "start_frame",
        "end_frame",
        "identities",
        "detail",
        "frames_extracted",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in events:
            writer.writerow(
                {
                    "video": e.video,
                    "event": e.event,
                    "start_frame": e.start_frame,
                    "end_frame": e.end_frame,
                    "identities": e.identities,
                    "detail": e.detail,
                    "frames_extracted": ";".join(e.frames_extracted),
                    "note": e.note,
                }
            )
    logger.info("Wrote %d events to %s", len(events), path)


@click.command()
@click.argument("pose_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("video_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--max-flicker",
    type=click.IntRange(min=0),
    default=30,
    show_default=True,
    help="Maximum number of flicker episodes to extract frames for (one per video at most).",
)
@click.option(
    "--max-extra",
    type=click.IntRange(min=0),
    default=100,
    show_default=True,
    help="Maximum number of extra-detection events to extract frames for.",
)
@click.option(
    "--max-extra-per-video",
    type=click.IntRange(min=0),
    default=3,
    show_default=True,
    help="Maximum number of extra-detection events to extract frames for per video.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Only scan the first N pose files (sorted by name).",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=max(1, (os.cpu_count() or 2) - 2),
    show_default=True,
    help="Number of worker processes.",
)
def main(
    pose_dir: Path,
    video_dir: Path,
    output_dir: Path,
    max_flicker: int,
    max_extra: int,
    max_extra_per_video: int,
    limit: int | None,
    workers: int,
) -> None:
    """Flag likely segmentation and tracking failures in POSE_DIR, reading VIDEO_DIR."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    console = Console()
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_files = sorted(pose_dir.glob(f"*{POSE_SUFFIX}"))[:limit]
    if not pose_files:
        raise click.ClickException(f"No *{POSE_SUFFIX} files in {pose_dir}")

    console.print(f"Scanning {len(pose_files)} pose files...")
    with Pool(workers) as pool:
        jobs = [(pose_file, video_dir) for pose_file in pose_files]
        events = [e for result in pool.imap_unordered(_detect, jobs) for e in result]
    cap_events(events, "flicker", per_video=1, overall=max_flicker)
    cap_events(events, "extra", per_video=max_extra_per_video, overall=max_extra)

    by_video: dict[str, list[FailureEvent]] = {}
    for event in events:
        by_video.setdefault(event.video, []).append(event)
    jobs = []
    for video_name, video_events in by_video.items():
        frames = sorted({f for e in video_events for f in e.key_frames})
        if not frames:
            continue
        video = video_dir / video_name
        if not video.exists():
            for e in video_events:
                e.note = "video not found"
            continue
        jobs.append((video_events[0].pose_file, video, frames, output_dir))

    console.print(f"Extracting frames from {len(jobs)} videos...")
    with Pool(workers) as pool:
        saved = dict(zip([job[1].name for job in jobs], pool.map(_extract, jobs), strict=True))

    for event in events:
        done = saved.get(event.video, set())
        stem = Path(event.video).stem
        event.frames_extracted = [f"{stem}_{f:05d}.png" for f in event.key_frames if f in done]
    events.sort(key=lambda e: (e.video, e.start_frame, e.event))
    _write_csv(events, output_dir / "events.csv")

    table = Table(title=f"{len(events)} events in {len(by_video)} of {len(pose_files)} videos")
    table.add_column("event")
    table.add_column("events", justify="right")
    table.add_column("with frames", justify="right")
    totals = Counter(e.event for e in events)
    extracted = Counter(e.event for e in events if e.frames_extracted)
    for name in ("fragment", "flicker", "extra", "short"):
        table.add_row(name, str(totals[name]), str(extracted[name]))
    console.print(table)
    raw_frames = sum(len(v) for v in saved.values())
    console.print(f"Wrote {raw_frames} raw frames and {raw_frames} overlay frames to {output_dir}")
    missing = sum(1 for v in by_video.values() if v[0].note == "video not found")
    if missing:
        console.print(f"[yellow]{missing} videos not found in {video_dir}[/yellow]")


if __name__ == "__main__":
    main()
