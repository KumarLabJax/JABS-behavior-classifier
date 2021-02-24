import argparse
import cv2
import h5py
import os
import random

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch-file',
        help='path to the file that is a new-line separated'
             ' list of all videos to process',
        required=True,
    )
    parser.add_argument(
        '--root-dir',
        help='the root directory. All paths given in the batch files are relative to this root',
        required=True,
    )
    parser.add_argument(
        '--out-dir',
        help='output directory. The videos and pose files for sampled intervals are saved to this dir',
        required=True,
    )
    parser.add_argument(
        '--out-frame-count',
        help='this defines how many frames to save. Assuming 30fps a value of 1800 corresponds to one minute',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--start-frame',
        help='this argument specifies which frame we start at. If this option is not specified we randomly select'
             ' a start frame from the video.',
        required=False,
        type=int,
    )
    parser.add_argument(
        '--pose-version',
        help='give the integer version number that should be used for pose',
        default=2,
        type=int,
        choices=(2, 3),
    )
    parser.add_argument(
        '--only-pose',
        help='if specified this option will sample pose data and exclude video from output',
        action='store_true',
    )

    args = parser.parse_args()

    if args.pose_version == 2:
        pose_suffix = '_pose_est_v2.h5'
    elif args.pose_version == 3:
        pose_suffix = '_pose_est_v3.h5'
    else:
        raise NotImplementedError('pose version not implemented: ' + str(args.pose_version))

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.batch_file) as batch_file:
        for line in batch_file:
            vid_filename = line.strip()
            if vid_filename:
                print('Processing:', vid_filename)
                vid_path = os.path.join(args.root_dir, vid_filename)
                vid_path_root, _ = os.path.splitext(vid_path)
                pose_in_path = vid_path_root + pose_suffix

                with h5py.File(pose_in_path, 'r') as pose_in:
                    frame_count = pose_in['poseest']['confidence'].shape[0]

                    last_candidate_frame = frame_count - args.out_frame_count
                    if last_candidate_frame <= 0:
                        print('WARNING: {} skipped because it only contains {} frames'.format(
                            vid_filename,
                            frame_count,
                        ))
                        continue

                    if args.start_frame is None:
                        out_start_frame_index = random.randrange(last_candidate_frame)
                    else:
                        out_start_frame_index = args.start_frame - 1

                    vid_out_filename = vid_filename.replace('/', '+').replace('\\', '+')
                    vid_out_path = os.path.join(args.out_dir, vid_out_filename)
                    vid_out_path_root, _ = os.path.splitext(vid_out_path)
                    vid_out_path = vid_out_path_root + '_' + str(out_start_frame_index + 1) + '.avi'
                    pose_out_path = vid_out_path_root + '_' + str(out_start_frame_index + 1) + pose_suffix

                    with h5py.File(pose_out_path, 'w') as pose_out:
                        start = out_start_frame_index
                        stop = start + args.out_frame_count
                        pose_out['poseest/points'] = pose_in['poseest/points'][start:stop, ...]
                        pose_out['poseest/confidence'] = pose_in['poseest/confidence'][start:stop, ...]
                        if 'instance_count' in pose_in['poseest']:
                            pose_out['poseest/instance_count'] = pose_in['poseest/instance_count'][start:stop, ...]
                        if 'instance_embedding' in pose_in['poseest']:
                            pose_out['poseest/instance_embedding'] = pose_in['poseest/instance_embedding'][start:stop, ...]
                        if 'instance_track_id' in pose_in['poseest']:
                            pose_out['poseest/instance_track_id'] = pose_in['poseest/instance_track_id'][start:stop, ...]
                        if 'version' in pose_in['poseest'].attrs:
                            pose_out['poseest'].attrs['version'] = pose_in['poseest'].attrs['version']

                    cap = None
                    writer = None

                    if not args.only_pose:
                        try:
                            cap = cv2.VideoCapture(vid_path)
                            if not cap.isOpened():
                                print('WARNING: failed to open {}'.format(vid_filename))
                                continue

                            cap.set(cv2.CAP_PROP_POS_FRAMES, out_start_frame_index)
                            if not cap.isOpened():
                                print('WARNING: failed to seek to start frame {}'.format(vid_filename))
                                continue

                            writer = cv2.VideoWriter(
                                vid_out_path,
                                cv2.VideoWriter_fourcc(*"MJPG"),
                                30,
                                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                            )
                            for _ in range(args.out_frame_count):

                                if not cap.isOpened():
                                    print('WARNING: {} ended prematurely'.format(vid_filename))
                                    break

                                ret, frame = cap.read()
                                if ret:
                                    writer.write(frame)
                                else:
                                    print('WARNING: {} ended prematurely'.format(vid_filename))
                                    break

                        finally:
                            if cap is not None:
                                cap.release()
                            if writer is not None:
                                writer.release()


if __name__ == "__main__":
    main()
