# DAX3 Envision MMPose Networks

Custom mouse pose + detection workflows built on MMPose/MMDetection, with training configs, small utility modules, and a set of inference/visualization scripts.

## What is in this repo

- `configs/custom_pose/`: source-of-truth training configs for pose (top-down + bottom-up) and detector.
- `src/train_custom_pose.py`: thin wrapper around `mmengine.Runner.from_cfg()` that keeps `src/` on `PYTHONPATH` and optionally renders dataset samples.
- `src/simple_pose/`: small custom components
  - `AssociativeEmbeddingHeadNoKptWeight` (bottom-up head variant)
  - `PackPoseInputsWithAE` (keeps AE labels)
  - `SingleClassAPMetric` (single-class detection AP)
  - `visualize_dataset_samples` (dataset sanity-check visuals)
- `scripts/`: data conversion, inference pipelines, rendering, PCK evaluation

## Training

Use the custom configs in `configs/custom_pose/` with the wrapper script so `simple_pose` is importable.

```bash
# Pose training (top-down)
apptainer run --nv mmpose.sif python src/train_custom_pose.py --config configs/custom_pose/topdown_resnet18.py

# Pose training (top-down, EfficientNetV2-S backbone)
apptainer run --nv mmpose.sif python src/train_custom_pose.py --config configs/custom_pose/topdown_efficientnetv2_s.py

# Pose training (bottom-up)
apptainer run --nv mmpose.sif python src/train_custom_pose.py --config configs/custom_pose/bottomup_hrnet_w32.py

# Detection training (mouse detector)
apptainer run --nv mmpose.sif python src/train_custom_pose.py --config configs/custom_pose/mouse_detector_retinanet.py

# Detection training (EfficientNetV2-S backbone)
apptainer run --nv mmpose.sif python src/train_custom_pose.py --config configs/custom_pose/mouse_detector_retinanet_efficientnetv2_s.py
```

### SLURM wrappers

If you're training on SLURM, there are `sbatch` wrappers under `scripts/`:

```bash
sbatch scripts/train_topdown_resnet18.sh
sbatch scripts/train_topdown_efficientnetv2_s.sh
sbatch scripts/train_bottomup_hrnet_w32.sh
sbatch scripts/train_mouse_detector_retinanet.sh
sbatch scripts/train_mouse_detector_retinanet_efficientnetv2_s.sh
```

If GPU utilization is low, start by increasing `train_dataloader.num_workers` and `train_dataloader.batch_size` in the pose config(s). The defaults here are tuned for better throughput on a local-disk dataset.

### Running without `train_custom_pose.py`

You can run the upstream OpenMMLab entrypoints directly, but you still need `simple_pose` importable because the base configs set `custom_imports = dict(imports=["simple_pose"], ...)`. Easiest is to add `src/` to `PYTHONPATH`:

```bash
# MMDetection (detector)
apptainer run --nv mmpose.sif bash -lc 'PYTHONPATH=src python -m mmdet.tools.train configs/custom_pose/mouse_detector_retinanet.py'

# MMPose (pose)
apptainer run --nv mmpose.sif bash -lc 'PYTHONPATH=src python -m mmpose.tools.train configs/custom_pose/topdown_resnet18.py'
```

### Monitoring

TensorBoard runs are under `runs/` by default:

```bash
# from this directory (ml/notebooks/keith/dax3-mmpose/)
tensorboard --logdir runs
```

## Data conversion

Convert custom JSON annotations into COCO keypoint format using `scripts/convert_custom_pose_to_coco.py`. This example converts train/val/test splits and strips the `s3://` prefix while adding a local image prefix.

```bash
# prefix_dir=/flashscratch/sheppk/hydra-label-cache
prefix_dir="$HOME/datasets/hydra-label-cache"
base_path="${prefix_dir}/dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail"
for split in train val test
do
  input_file="${base_path}_${split}.json"
  output_file="${input_file%.json}_coco.json"
  python ml/notebooks/keith/dax3-mmpose/scripts/convert_custom_pose_to_coco.py \
      --strip-prefix s3:// \
      --image-prefix "${prefix_dir}/" \
      "$input_file" "$output_file"
done
```

## Inference pipelines

This repo has three detector -> top-down pose inference entrypoints:

- `scripts/img_infer_detector_then_pose.py`: single image, renders an output image via the model visualizer.
- `scripts/vid_infer_detector_then_pose.py`: video, emits JSONL or parquet per frame.
- `scripts/coco_infer_detector_then_pose.py`: COCO dataset, writes JSONL (one record per image).

### Single-image inference

Run a detector then a pose model on a single image and write a rendered output.

```bash
APPTAINERENV_PYTHONPATH=src apptainer run --nv mmpose.sif \
  python scripts/img_infer_detector_then_pose.py \
  ~/vlcsnap-2026-01-07-14h50m01s891.png \
  --det-config configs/custom_pose/mouse_detector_retinanet.py \
  --det-checkpoint runs/mouse_detector/best_single_class_ap_AP@0.75_epoch_23.pth \
  --pose-config configs/custom_pose/topdown_resnet18.py \
  --pose-checkpoint runs/custom_pose/best_PCK_epoch_45.pth \
  --out-file ~/vlcsnap-2026-01-07-14h50m01s891_out.png
```

### Video inference with parquet output

Run the detector + pose model on a video, emit parquet, then render keypoints back onto the video.

```bash
#det_checkpoint=runs/mouse_detector/best_single_class_ap_AP@0.75_epoch_23.pth
#pose_checkpoint=runs/custom_pose/best_PCK_epoch_45.pth
det_checkpoint=runs/mouse_detector/best_single_class_ap_AP@0.75_epoch_28.pth
pose_checkpoint=runs/custom_pose/best_PCK_epoch_60.pth

APPTAINERENV_PYTHONPATH=src apptainer run --nv mmpose.sif \
  python scripts/vid_infer_detector_then_pose.py \
  ~/datasets/tuans-lixit-vid/testVideo_lixit/cage_2459.2025-02-11.03.19.top.mp4 \
  --det-config configs/custom_pose/mouse_detector_retinanet.py \
  --det-checkpoint "$det_checkpoint" \
  --pose-config configs/custom_pose/topdown_resnet18.py \
  --pose-checkpoint "$pose_checkpoint" \
  --max-instances 3 \
  --batch-size 4 \
  --out-parquet ~/datasets/tuans-lixit-vid/testVideo_lixit/cage_2459.2025-02-11.03.19.top.parquet

APPTAINERENV_PYTHONPATH=src apptainer run --nv mmpose.sif \
  python scripts/render_pose_on_video.py \
  ~/datasets/tuans-lixit-vid/testVideo_lixit/cage_2459.2025-02-11.03.19.top.mp4 \
  ~/datasets/tuans-lixit-vid/testVideo_lixit/cage_2459.2025-02-11.03.19.top.parquet \
  ~/datasets/tuans-lixit-vid/testVideo_lixit/cage_2459.2025-02-11.03.19.top.pose_render.mp4 \
  --draw-bbox \
  --exclude-points 4
```

### Video inference with JSONL output

Same pipeline, but capture JSONL to a file and render from JSONL.

```bash
APPTAINERENV_PYTHONPATH=src apptainer run --nv mmpose.sif \
  python scripts/vid_infer_detector_then_pose.py \
  ~/datasets/tuans-lixit-vid/testVideo_lixit/cage_2459.2025-02-11.03.19.top.mp4 \
  --det-config configs/custom_pose/mouse_detector_retinanet.py \
  --det-checkpoint runs/mouse_detector/best_single_class_ap_AP@0.75_epoch_23.pth \
  --pose-config configs/custom_pose/topdown_resnet18.py \
  --pose-checkpoint runs/custom_pose/best_PCK_epoch_45.pth \
  --max-instances 3 \
  --batch-size 4 \
  > ~/datasets/tuans-lixit-vid/testVideo_lixit/cage_2459.2025-02-11.03.19.top.pose.jsonl

APPTAINERENV_PYTHONPATH=src apptainer run --nv mmpose.sif \
  python scripts/render_pose_on_video.py \
  ~/datasets/tuans-lixit-vid/testVideo_lixit/cage_2459.2025-02-11.03.19.top.mp4 \
  ~/datasets/tuans-lixit-vid/testVideo_lixit/cage_2459.2025-02-11.03.19.top.pose.jsonl \
  ~/datasets/tuans-lixit-vid/testVideo_lixit/cage_2459.2025-02-11.03.19.top.pose_render.mp4 \
  --draw-bbox \
  --exclude-points 4
```

### Video inference (alternate dataset)

Example with a different input video and output paths.

```bash
#det_checkpoint=runs/mouse_detector/best_single_class_ap_AP@0.75_epoch_23.pth
#pose_checkpoint=runs/custom_pose/best_PCK_epoch_45.pth
det_checkpoint=runs/mouse_detector/best_single_class_ap_AP@0.75_epoch_28.pth
pose_checkpoint=runs/custom_pose/best_PCK_epoch_60.pth

APPTAINERENV_PYTHONPATH=src apptainer run --nv mmpose.sif \
  python scripts/vid_infer_detector_then_pose.py \
  ~/temp/cage_1827.2024-05-31.4.7.mp4 \
  --det-config configs/custom_pose/mouse_detector_retinanet.py \
  --det-checkpoint "$det_checkpoint" \
  --pose-config configs/custom_pose/topdown_resnet18.py \
  --pose-checkpoint "$pose_checkpoint" \
  --max-instances 3 \
  --batch-size 4 \
  --out-parquet ~/temp/cage_1827.2024-05-31.4.7.parquet

APPTAINERENV_PYTHONPATH=src apptainer run --nv mmpose.sif \
  python scripts/render_pose_on_video.py \
  ~/temp/cage_1827.2024-05-31.4.7.mp4 \
  ~/temp/cage_1827.2024-05-31.4.7.parquet \
  ~/temp/cage_1827.2024-05-31.4.7_render.mp4 \
  --draw-bbox \
  --exclude-points 4
```

### COCO inference + PCK evaluation

Run detector -> pose on a COCO dataset, write JSONL, then compute PCK against the COCO GT.

```bash
#det_checkpoint=runs/mouse_detector/best_single_class_ap_AP@0.75_epoch_23.pth
#pose_checkpoint=runs/custom_pose/best_PCK_epoch_45.pth
det_checkpoint=runs/mouse_detector/best_single_class_ap_AP@0.75_epoch_28.pth
pose_checkpoint=runs/custom_pose/best_PCK_epoch_60.pth

APPTAINERENV_PYTHONPATH=src apptainer run --nv mmpose.sif \
  python scripts/coco_infer_detector_then_pose.py \
  ~/datasets/hydra-label-cache/dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_test_coco.json \
  --det-config configs/custom_pose/mouse_detector_retinanet.py \
  --det-checkpoint "$det_checkpoint" \
  --pose-config configs/custom_pose/topdown_resnet18.py \
  --pose-checkpoint "$pose_checkpoint" \
  --max-instances 3 \
  --out-jsonl 2025-09-17d_no-tail_test_inference.jsonl

APPTAINERENV_PYTHONPATH=src apptainer run --nv mmpose.sif \
  python scripts/compute_pck_from_coco_jsonl.py \
  2025-09-17d_no-tail_test_inference.jsonl \
  ~/datasets/hydra-label-cache/dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_test_coco.json \
  --pck-thr 0.05 \
  --norm max \
  --min-iou 0.1 \
  --exclude-points 4
```

### Hydra comparison run (alternate model)

Example running an alternate model to produce JSONL, then computing PCK with the same evaluator.

```bash
APPTAINERENV_PYTHONPATH=src apptainer run --nv hydra-dev.sif \
  python src/daxml/hydra/inference/inference_coco_jsonl.py \
  --max-instances 3 \
  --keypoint_threshold 0.0 \
  --out-jsonl 2025-09-17d_no-tail_test_inference_hydra-0-kp-thresh.jsonl \
  --hydra_checkpoint ~/temp/analytical_validation_2025-09-17d_mouse-cluster_epoch\=163-val_loss\=5.346.ckpt \
  --hydra_config ~/temp/db_trainer_2025-09-17d_no-dravet_no-tail_huddle.yaml \
  --no-trt \
  ~/datasets/hydra-label-cache/dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_test_coco.json

APPTAINERENV_PYTHONPATH=src apptainer run --nv mmpose.sif \
  python scripts/compute_pck_from_coco_jsonl.py \
  ~/projects/murine-mmpose-impl/ml-common/daxml/2025-09-17d_no-tail_test_inference_hydra-0-kp-thresh.jsonl \
  ~/datasets/hydra-label-cache/dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_test_coco.json \
  --pck-thr 0.05 \
  --norm max \
  --min-iou 0.1 \
  --exclude-points 4
```

## Rendering utilities

`scripts/render_pose_on_video.py` draws predicted keypoints (JSONL or parquet) back onto a video. It supports per-keypoint thresholds, optional bounding boxes, and excluding points from rendering.

```bash
APPTAINERENV_PYTHONPATH=src apptainer run --nv mmpose.sif \
  python scripts/render_pose_on_video.py \
  ~/datasets/tuans-lixit-vid/testVideo_lixit/cage_2459.2025-02-11.03.19.top.mp4 \
  ~/datasets/tuans-lixit-vid/testVideo_lixit/cage_2459.2025-02-11.03.19.top.parquet \
  ~/datasets/tuans-lixit-vid/testVideo_lixit/cage_2459.2025-02-11.03.19.top.pose_render.mp4 \
  --draw-bbox \
  --exclude-points 4
```

## PCK evaluation

`scripts/compute_pck_from_coco_jsonl.py` computes PCK from `scripts/coco_infer_detector_then_pose.py` JSONL outputs against COCO ground truth. It supports different normalization modes and excluding keypoint indices.

```bash
APPTAINERENV_PYTHONPATH=src apptainer run --nv mmpose.sif \
  python scripts/compute_pck_from_coco_jsonl.py \
  2025-09-17d_no-tail_test_inference.jsonl \
  ~/datasets/hydra-label-cache/dax-ml-datasets/datasets-murine/data_versions/2025-09-17d_dax3_v7_chkpt/2025-09-17d_no-tail_test_coco.json \
  --pck-thr 0.05 \
  --norm max \
  --min-iou 0.1 \
  --exclude-points 4
```

## Container build + runtime notes

These examples show how the MMPose Apptainer image is built and how to sanity-check it.

```bash
# from this directory (ml/notebooks/keith/dax3-mmpose/)
docker build -t mmpose-base:latest .

# rebuild the Apptainer image (use --force to overwrite)
apptainer build --force mmpose.sif mmpose.def
```

```bash
apptainer run --nv mmpose.sif python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.enabled)"
apptainer run --nv mmpose.sif python -c "import torch; print(torch.cuda.is_available())"
apptainer run --cleanenv --no-home mmpose.sif
apptainer run --cleanenv --home /tmp/sing-home mmpose.sif
apptainer shell --nv --cleanenv --no-home mmpose.sif
docker run -it --rm mmpose-base:latest /bin/bash
```

```bash
apptainer run --nv mmpose.sif python src/train_custom_pose.py

apptainer run --nv mmpose.sif python src/train_custom_pose.py --config configs/custom_pose/mouse_detector_retinanet.py
```

## Config notes

For pose configs, both top-down and bottom-up use `PCKAccuracy(thr=0.05, norm_item="bbox")` so the normalization is consistent across the two modes.

Both pose configs also log COCO keypoint metrics via `CocoMetric` during val/test, so you can compare `coco/AP` alongside PCK.

## Legacy notes / reference snippets

These are preserved reference snippets from prior iterations, kept verbatim.

```bash
# Detector → Top-down pose inference (WIP)
apptainer run --nv mmpose.sif python scripts/infer_detector_then_pose.py \
  path/to/image.png \
  --det-config configs/custom_pose/mouse_detector_retinanet.py \
  --det-checkpoint path/to/detector.pth \
  --pose-config configs/custom_pose/topdown_resnet18.py \
  --pose-checkpoint path/to/pose.pth \
  --out-file runs/infer_vis.png
```

```
removed from dataset.py:

try:
    from mmdet.registry import DATASETS as MMDET_DATASETS
except ImportError:  # pragma: no cover
    MMDET_DATASETS = None

...

if MMDET_DATASETS is not None:
    MMDET_DATASETS.register_module()(JsonKeypointDataset)
```



NOTE TO SELF: use softlink for: data_root = "/flashscratch/sheppk/hydra-label-cache"

for:

* ml/notebooks/keith/dax3-mmpose/configs/custom_pose/bottomup_hrnet_w32.py
* ml/notebooks/keith/dax3-mmpose/configs/custom_pose/mouse_detector_retinanet.py
* ml/notebooks/keith/dax3-mmpose/configs/custom_pose/topdown_resnet18.py
