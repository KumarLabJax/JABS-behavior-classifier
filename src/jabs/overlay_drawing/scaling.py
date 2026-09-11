"""Marker scaling for overlays drawn at native video resolution.

The on-screen overlays draw at the display's scale, where a fixed marker size reads
well. The frame and video exports draw at the video's native resolution instead, which
varies from 480x480 to 1080p across the footage JABS is used on, so their markers are
scaled from the frame size here. Keeping the one scale factor in a single module means
the pose skeleton, the label markers and the caption all grow together.
"""

# Frame size the base marker sizes are calibrated against, taken from 800x800
# open-field footage where a 3px keypoint radius reads well.
_REFERENCE_FRAME = 800


def native_overlay_scale(width: int, height: int) -> float:
    """Return the factor to scale overlay markers by for a frame of this size.

    Markers grow with the frame so the overlay stays legible across resolutions, but
    they grow *sub-linearly* - as the square root of the frame's larger dimension.

    Scaling linearly (the original behavior) keeps a marker at a constant fraction of
    the frame, which sounds right but is not what the eye judges. Measured on real
    footage: a mouse in 1080p home-cage video is about four times longer in pixels
    than one in 800x800 open-field video, so linear scaling produced 16px dots that
    covered the animal, while the same formula gave a well-judged 6px dot on the
    smaller frame. What reads badly is a marker's absolute size, not its share of the
    frame.

    Sizing markers from the animal's own extent was considered and rejected: matching
    the 800x800 proportions (dot diameter around 6% of body length) would make the
    1080p dots *larger* still, at roughly 24px.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Multiplier to apply to a marker's base (800x800) size.
    """
    return (max(width, height, 1) / _REFERENCE_FRAME) ** 0.5
