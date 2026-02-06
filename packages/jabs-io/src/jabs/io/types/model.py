from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about the model used for inference.

    Attributes:
        checkpoint_path: Path to the model checkpoint.
        backbone: Backbone identifier (e.g., mobilenetv3_large_100).
        num_keypoints: Number of keypoints predicted by the model.
        input_size: Input size used for inference as (H, W).
        output_stride: Output stride for heatmap decoding.
        decode_use_dark: Whether DARK refinement was used.
        decode_sigma: Sigma value for DARK refinement.
    """

    checkpoint_path: str
    backbone: str
    num_keypoints: int
    input_size: tuple[int, int]
    output_stride: int
    decode_use_dark: bool = True
    decode_sigma: float = 2.0
