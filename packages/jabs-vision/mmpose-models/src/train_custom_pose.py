from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import strftime

import mmpose.codecs
import mmpose.datasets.transforms  # noqa: F401
from mmengine.config import Config
from mmengine.runner import Runner


def compute_slurm_id() -> str | None:
    """Compute SLURM job ID from environment variables.
    
    Returns:
        str | None: The SLURM job ID or array job ID, or None if not running under SLURM.
    """
    slurm_job_id = os.getenv("SLURM_JOB_ID", "").strip()
    slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID", "").strip()
    slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID", "").strip()

    if slurm_array_job_id and slurm_array_task_id:
        return f"{slurm_array_job_id}_{slurm_array_task_id}"
    elif slurm_job_id:
        return slurm_job_id
    else:
        return None


def main() -> None:
    """Train a pose model using a config file."""
    parser = argparse.ArgumentParser(description="Train a pose model using a config file")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the training config",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(str(args.config))

    # update the configuration's working directory to include the configuration
    # name and a timestamp or SLURM ID for better organization of training runs
    config_stem = args.config.stem
    slurm_id = compute_slurm_id()
    run_id = slurm_id if slurm_id else strftime("%Y%m%d-%H%M%S")
    work_dir = Path("runs") / config_stem / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir = str(work_dir)

    print(
        f"update work_dir={work_dir}. Logs and checkpoints will be saved there."
        f" Starting training...")

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
