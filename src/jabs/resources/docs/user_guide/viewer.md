# JABS Viewer

`jabs-viewer` is a standalone player for checking pose estimation output. It plays a
video with its pose, segmentation and bounding boxes drawn on top, or plays a pose
file on its own when there is no video. It uses the same video player as the main JABS application
but does not need a JABS project: open a file and watch.

## Launching the Viewer

```bash
jabs-viewer [file]
```

- `file`: Optional. A video file (`.mp4` or `.avi`) or a JABS pose file (`.h5`) to
  open on startup. Files ending in `.h5` are opened as pose files; anything else is
  opened as a video.

Without an argument the viewer starts empty. Open a file from the **File** menu or the
toolbar.

## Opening a Video

Use **File→Open Video...** (or the **Open Video** toolbar button). If a JABS pose file
named `<video name>_pose_est_v<N>.h5` is in the same directory as the video, it is
opened as well and drawn over the video. When more than one version is present, the
highest version is used.

A video with no pose file beside it plays without overlays, and the status bar
reports that no pose file was found. If the pose file exists but cannot be opened,
or has a different number of frames than the video, the viewer says so and plays the
video without overlays.

## Opening a Pose File Without Video

Use **File→Open Pose File...** (or the **Open Pose File** toolbar button) to play a
pose file on its own. The pose is drawn over a solid dark gray background at 30
fps, since pose files do not record the frame rate of the video they came from.

### Frame Size

Pose files do not record the size of the video they were produced from, so the
viewer estimates it from the pose coordinates: it takes the largest x and y of the
animals' keypoints and the arena landmarks (such as the arena corners), and rounds
each up to the next multiple of 100 pixels. For pose files that include arena corners
(version 5 and newer) this normally matches the original video, e.g. 800 x 800.
Older files without corners can come out smaller than the original video, since the
estimate only covers where the animals went. The status bar shows the frame size in
use and whether it was estimated or set by you.

To correct the frame size, use **View→Set Frame Size...**. The dialog starts at the
current size and shows the estimate for reference; **Use Estimate** restores it.
Applying a new size keeps the current frame position. This option is only available
while a pose file is open without a video; a video's frames are always the video's
own size.

## Overlays

All animals are drawn the same way. There is no selected animal in the viewer.

- **Overlay Pose** (`P`): Show or hide the pose keypoints and skeleton for every
  animal.
- **Overlay Segmentation** (`S`): Show or hide the segmentation contours. This is
  only available when the pose file contains segmentation data (version 6 or newer,
  and generated with segmentation, which is optional).
- **Overlay Bounding Boxes** (`B`): Show or hide each animal's bounding box, labeled
  with its identity. This is only available when the pose file contains bounding
  boxes (version 8 or newer, and generated with bounding boxes, which is optional).

All are on by default and can be toggled from the **View** menu or the toolbar.
They are unavailable when no pose file is open.

## Exporting

### Export Frame

**File→Export Frame...** (Ctrl+E / ⌘E) saves the current frame as a PNG image at the
video's native resolution. With a video open, the frame is saved without overlays,
and the **Also save a copy with the displayed overlays** checkbox saves a second
image, named `<name>-overlay.png`, with the overlays currently switched on in the
**View** menu. With a pose file open without a video, only the frame with the
overlays is saved, since the background alone is not worth a file. If the video is
playing it is paused first.

### Export Video with Overlays

**File→Export Video with Overlays...** writes an MP4 copy of the video with the
overlays drawn on every frame. A dialog first asks which overlays to include (pose,
segmentation and bounding boxes), starting from those currently shown; overlays the
pose file does not have are disabled, with the reason as a tooltip. You then choose
where to save the file. The export runs in the background with a progress dialog
and can be cancelled, which removes the partial file.

With a pose file open without a video, the pose is drawn over the same dark gray
background, at the frame size in use, at 30 fps. Exporting a video needs a pose
file; it is unavailable for a video with no pose.

## Video Controls

The player has the same controls as the main JABS window: play/pause button,
position slider, previous/next frame buttons, and the controls that appear when the
mouse is over the video (playback speed, cropping, brightness and contrast).

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Space` | Play / pause |
| `←` / `→` | Previous / next frame |
| `P` | Toggle pose overlay |
| `S` | Toggle segmentation overlay |
| `B` | Toggle bounding box overlay |
| Ctrl+O / ⌘O | Open video |
| Ctrl+E / ⌘E | Export frame |
