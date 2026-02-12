from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import strftime

import mmpose.codecs  # noqa: F401
import mmpose.datasets.transforms  # noqa: F401
from mmengine.config import Config
from mmengine.runner import Runner


def compute_slurm_id() -> str | None:
    slurm_job_id = os.getenv("SLURM_JOB_ID", "").strip()
    slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID", "").strip()
    slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID", "").strip()

    if slurm_array_job_id and slurm_array_task_id:
        return f"{slurm_array_job_id}_{slurm_array_task_id}"
    elif slurm_job_id:
        return slurm_job_id
    else:
        return None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pose model using a config file")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the training config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_stem = args.config.stem
    slurm_id = compute_slurm_id()
    if slurm_id:
        work_dir = Path("runs") / config_stem / slurm_id
    else:
        work_dir = Path("runs") / config_stem
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"work_dir={work_dir}")
    cfg = Config.fromfile(str(args.config))
    cfg.work_dir = str(work_dir)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
