# Labeling

This section describes how a user can add or remove labels. Labels are always applied to the subject mouse and the current subject can be changed at any time. A common way to approach labeling is to scan through the video for the behavior of interest, and then when the behavior is observed select the mouse that is showing the behavior. Scan to the start of the behavior, and begin selecting frames. Scan to the end of the behavior to select all of the frames that belong to the bout, and click the label button.

## Selecting Frames

When "Select Mode" is activated, JABS begins a new selection starting at that frame. The current selection is from the selection start frame through the current frame. Applying a label, or removing labels from the selection clears the current selection and leaves "Select Mode".

The current selection range is shown on the "Manual Labels" display:

<img src="imgs/selecting_frames.png" alt="Selecting Frames" width=900 />

Clicking the "Select Frames" button again or pressing the Escape key will deselect the frames and leave select mode without making a change to the labels.

## Applying Labels

The label **Behavior** button will mark the selected interval of frames as showing the current behavior. The label **Not Behavior** button will mark all the frames in the selected interval as not showing the behavior. The **New Timeline Annotation** button will open the Timeline Annotation Editor dialog to create a new timeline annotation for the selected interval. Finally, the "Clear Labels" button will remove all labels from the currently selected frames.

## Timeline Annotations

Timeline annotations mark frame intervals that are not tied to a behavior label. They are never used for training and can apply to the entire video or to a specific animal (via an identity). Each annotation includes a short tag, an optional description, and a display color. Tags appear as overlays in the video and can be searched (with search hits highlighted in the label timeline), making it easy to flag edge cases, highlight areas of disagreement for review, or note uncertainty and poor pose quality.

### Where They're Stored

Each video's annotations are saved to `jabs/annotations/<video_name>.json` inside the project directory.

Example annotation file (other top-level fields omitted for clarity):

```json
{
  "annotations": [
    {
      "start": 100,
      "end": 200,
      "tag": "identity",
      "identity": 0,
      "color": "#ff0000",
      "description": "identity is wrong"
    },
    {
      "start": 150,
      "end": 250,
      "tag": "obstructed",
      "identity": null,
      "color": "#0000ff",
      "description": "view is obstructed"
    }
  ]
}
```

### Fields

- start (int): first frame index (inclusive).
- end (int): last frame index (inclusive).
- tag (str): short label for quick identification (see Tag rules).
- identity (int | null): optional animal identity; if omitted or null, the annotation applies to the whole video.
- color (str): display color, e.g. `#RRGGBB` or an SVG color name.
- description (str, optional): free-text notes.

### Tag Rules

- Characters: letters, digits, underscores `_`, and hyphens `-`; **no whitespace**.
- Length: 1–32 characters.
- Matching/filtering is case-insensitive; display preserves your original casing.

### How They're Displayed

- Tag Overlay: the tag appears as a badge in the video.
- If identity is set, the tag follows that animal.
- If the pose file includes external identity mapping, the JABS identity is converted to the external ID for display.
- If pose is missing, the tag snaps to the upper-left corner and is prefixed with the identity (e.g., 1234: tag).
- Clicking a tag opens details.

## Identity Gaps

Identities can have gaps if the mouse becomes obstructed or the pose estimation failed for those frames. In the manual label visualization, these gaps are indicated with a pattern fill instead of the solid gray/orange/blue colors. In the predicted class visualization, the gaps are colored white (meaning no prediction exists for these frames).

<img src="imgs/identity_gaps.png" alt="Identity Gaps" width=900 />

Labels can be saved on frames where the identity is missing; however, these labels are excluded from classifier training. In the label timeline, labels applied to identity gaps are displayed with partial transparency, allowing the gap pattern fill to remain visible.

<img src="imgs/identity_gaps_with_label.png" alt="Identity Gaps with Labels" />

> **Note:** Gaps are excluded from classifier training but are still displayed in timelines. Labels applied in gap regions use partial transparency and won't bias the classifier.

## Keyboard Shortcuts

Keyboard controls are the fastest way to label behaviors. The most commonly used shortcuts are:

- **z, x, c** - Start selecting frames (if not in select mode)
- Modify selected frames:
  - **z** - Apply "behavior" label to selected frames.
  - **x** - Clear label from selected frames.
  - **c** - Apply "not behavior" label to selected frames.
- **← / →** - Navigate one frame backward/forward
- **↑ / ↓** - Skip 10 frames backward/forward
- **Esc** - Exit select mode without applying labels

For a complete list of all keyboard shortcuts, see the [Keyboard Shortcuts Reference](keyboard-shortcuts.md).
