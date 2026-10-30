# JABS Viewer

`jabs-viewer` is a standalone player for checking pose estimation output. It plays a
video with its pose and segmentation drawn on top, or plays a pose file on its own
when there is no video. It uses the same video player as the main JABS application
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

Both are on by default and can be toggled from the **View** menu or the toolbar.
They are unavailable when no pose file is open.

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
| Ctrl+O / ⌘O | Open video |
