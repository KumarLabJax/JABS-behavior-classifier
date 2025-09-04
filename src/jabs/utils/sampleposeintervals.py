import argparse
import os
import random

import cv2
import h5py

# Command line example of using this script:
#
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
#   python src/utils/sampleposeintervals.py \
#       --batch-file UCSD_Rotta_TS_v2.txt \
#       --root-dir "${share_root}" \
#       --out-dir UCSD_Rotta_TS_v2-intervals \
#       --out-frame-count 9000 \
#       --start-frame 54000 \
#       --pose-version 3
#
#   share_root='/media/sheppk/TOSHIBA EXT/rotta-data/UCSD_Rotta_TS_v2-vidcache'
#   python src/utils/sampleposeintervals.py \
#       --batch-file "${share_root}/batch.txt" \
#       --root-dir "${share_root}" \
#       --out-dir UCSD_Rotta_TS_v2-intervals-2021-05-25 \
#       --out-frame-count 9000 \
#       --start-frame 27000 \
#       --pose-version 3

#   python src/utils/sampleposeintervals.py \
#       --batch-file ~/projects/social-interaction/data/bxd-batch-early-morning-2021-06-09.txt \
#       --root-dir '/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#       --out-dir bxd-batch-early-morning-2021-06-09 \
#       --out-frame-count 9000 \
#       --start-frame 54000 \
#       --pose-version 3

#   python src/utils/sampleposeintervals.py \
#       --batch-file temp/B6J-and-BTBR-3M-strangers-4-day-rand-2021-05-24.txt \
#       --root-dir '/media/sheppk/TOSHIBA EXT/rotta-data/B6J-and-BTBR-3M-strangers-4-day-rand-2021-05-24' \
#       --out-dir B6J-and-BTBR-3M-strangers-4-day-rand-samples-2021-05-24 \
#       --out-frame-count 3600 \
#       --start-frame 6000 \
#       --pose-version 3

#   python src/utils/sampleposeintervals.py \
#       --batch-file temp/B6J-and-BTBR-3M-strangers-4-day-rand-2021-05-24.txt \
#       --root-dir '/media/sheppk/TOSHIBA EXT/rotta-data/B6J_and_BTBR_3M_stranger_4day_2021-07-20' \
#       --out-dir temp/B6J-and-BTBR-3M-strangers-4-day-rand-samples-2021-08-05 \
#       --out-frame-count 3600 \
#       --start-frame 6000 \
#       --only-pose \
#       --pose-version 4

#   rclone copy --transfers 4 --progress \
#       --include-from /home/sheppk/projects/behavior-classifier/temp/BTBR_3M_stranger_4day-subset-avi.txt \
#       "labdropbox:/KumarLab's shared workspace/VideoData/MDS_Tests/BTBR_3M_stranger_4day" \
#       /media/sheppk/TOSHIBA\ EXT/BTBR_3M_stranger_4day-2021-08-24
#   rclone copy --transfers 4 --progress \
#       --include-from /home/sheppk/projects/behavior-classifier/temp/BTBR_3M_stranger_4day-subset-pose.txt \
#       /home/sheppk/sshfs/winterproj/bgeuther/IdentityInfer/Data/BTBR_3M_stranger_4day \
#       /media/sheppk/TOSHIBA\ EXT/BTBR_3M_stranger_4day-2021-08-24
#   python src/utils/sampleposeintervals.py \
#       --batch-file /media/sheppk/TOSHIBA\ EXT/BTBR_3M_stranger_4day-2021-08-24/batch.txt \
#       --root-dir /media/sheppk/TOSHIBA\ EXT/BTBR_3M_stranger_4day-2021-08-24 \
#       --out-dir /media/sheppk/TOSHIBA\ EXT/BTBR_3M_stranger_4day-2021-08-24-samples \
#       --out-frame-count 3600 \
#       --start-frame 6000 \
#       --pose-version 4


def main():
    """sample pose intervals"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-file",
        help="path to the file that is a new-line separated list of all videos to process",
        required=True,
    )
    parser.add_argument(
        "--root-dir",
        help="the root directory. All paths given in the batch files are relative to this root",
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        help="output directory. The videos and pose files for sampled intervals are saved to this dir",
        required=True,
    )
    parser.add_argument(
        "--out-frame-count",
        help="this defines how many frames to save. Assuming 30fps a value of 1800 corresponds to one minute",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--start-frame",
        help="this argument specifies which frame we start at. If this option is not specified we randomly select"
        " a start frame from the video.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--pose-version",
        help="give the integer version number that should be used for pose",
        default=2,
        type=int,
        choices=(2, 3, 4, 5),
    )
    parser.add_argument(
        "--only-pose",
        help="if specified this option will sample pose data and exclude video from output",
        action="store_true",
    )

    args = parser.parse_args()

    if args.pose_version == 2:
        pose_suffix = "_pose_est_v2.h5"
    elif args.pose_version == 3:
        pose_suffix = "_pose_est_v3.h5"
    elif args.pose_version == 4:
        pose_suffix = "_pose_est_v4.h5"
    elif args.pose_version == 5:
        pose_suffix = "_pose_est_v5.h5"
    else:
        raise NotImplementedError("pose version not implemented: " + str(args.pose_version))

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.batch_file) as batch_file:
        for line in batch_file:
            vid_filename = line.strip()
            if vid_filename:
                print("Processing:", vid_filename)
                vid_path = os.path.join(args.root_dir, vid_filename)
                vid_path_root, _ = os.path.splitext(vid_path)
                pose_in_path = vid_path_root + pose_suffix

                if not args.only_pose and not os.path.isfile(vid_path):
                    print("WARNING: missing video path:", vid_path)
                    continue

                if not os.path.isfile(pose_in_path):
                    print("WARNING: missing pose path:", pose_in_path)
                    continue

                with h5py.File(pose_in_path, "r") as pose_in:
                    frame_count = pose_in["poseest"]["confidence"].shape[0]

                    last_candidate_frame = frame_count - args.out_frame_count
                    if last_candidate_frame <= 0:
                        print(
                            f"WARNING: {vid_filename} skipped because it only contains {frame_count} frames"
                        )
                        continue

                    if args.start_frame is None:
                        out_start_frame_index = random.randrange(last_candidate_frame)
                    else:
                        out_start_frame_index = args.start_frame - 1

                    vid_out_filename = vid_filename.replace("/", "+").replace("\\", "+")
                    vid_out_path = os.path.join(args.out_dir, vid_out_filename)
                    vid_out_path_root, _ = os.path.splitext(vid_out_path)
                    vid_out_path = (
                        vid_out_path_root + "_" + str(out_start_frame_index + 1) + ".avi"
                    )
                    pose_out_path = (
                        vid_out_path_root + "_" + str(out_start_frame_index + 1) + pose_suffix
                    )

                    with h5py.File(pose_out_path, "w") as pose_out:
                        # pose v2 stuff
                        start = out_start_frame_index
                        stop = start + args.out_frame_count
                        pose_out["poseest/points"] = pose_in["poseest/points"][start:stop, ...]
                        pose_out["poseest/confidence"] = pose_in["poseest/confidence"][
                            start:stop, ...
                        ]

                        # pose v3 stuff
                        if "instance_count" in pose_in["poseest"]:
                            pose_out["poseest/instance_count"] = pose_in["poseest/instance_count"][
                                start:stop, ...
                            ]
                        if "instance_embedding" in pose_in["poseest"]:
                            pose_out["poseest/instance_embedding"] = pose_in[
                                "poseest/instance_embedding"
                            ][start:stop, ...]
                        if "instance_track_id" in pose_in["poseest"]:
                            pose_out["poseest/instance_track_id"] = pose_in[
                                "poseest/instance_track_id"
                            ][start:stop, ...]

                        # pose v4 stuff
                        if "id_mask" in pose_in["poseest"]:
                            pose_out["poseest/id_mask"] = pose_in["poseest/id_mask"][
                                start:stop, ...
                            ]
                        if "identity_embeds" in pose_in["poseest"]:
                            pose_out["poseest/identity_embeds"] = pose_in[
                                "poseest/identity_embeds"
                            ][start:stop, ...]
                        if "instance_embed_id" in pose_in["poseest"]:
                            pose_out["poseest/instance_embed_id"] = pose_in[
                                "poseest/instance_embed_id"
                            ][start:stop, ...]
                        if "instance_id_center" in pose_in["poseest"]:
                            pose_out["poseest/instance_id_center"] = pose_in[
                                "poseest/instance_id_center"
                            ][:]

                        # v5 specific stuff
                        if "static_objects" in pose_in:
                            static_group = pose_out.create_group("static_objects")
                            for dataset in pose_in["static_objects"]:
                                static_group.create_dataset(
                                    dataset, data=pose_in["static_objects"][dataset]
                                )

                        # copy attributes
                        for attr in pose_in["poseest"].attrs:
                            pose_out["poseest"].attrs[attr] = pose_in["poseest"].attrs[attr]

                    cap = None
                    writer = None

                    if not args.only_pose:
                        try:
                            cap = cv2.VideoCapture(vid_path)
                            if not cap.isOpened():
                                print(f"WARNING: failed to open {vid_filename}")
                                continue

                            cap.set(cv2.CAP_PROP_POS_FRAMES, out_start_frame_index)
                            if not cap.isOpened():
                                print(f"WARNING: failed to seek to start frame {vid_filename}")
                                continue

                            writer = cv2.VideoWriter(
                                vid_out_path,
                                cv2.VideoWriter_fourcc(*"MJPG"),
                                30,
                                (
                                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                ),
                            )
                            for _ in range(args.out_frame_count):
                                if not cap.isOpened():
                                    print(f"WARNING: {vid_filename} ended prematurely")
                                    break

                                ret, frame = cap.read()
                                if ret:
                                    writer.write(frame)
                                else:
                                    print(f"WARNING: {vid_filename} ended prematurely")
                                    break

                        finally:
                            if cap is not None:
                                cap.release()
                            if writer is not None:
                                writer.release()


if __name__ == "__main__":
    main()
