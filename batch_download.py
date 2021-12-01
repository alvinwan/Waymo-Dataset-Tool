from __future__ import print_function
import glob
import os
import argparse
from convert_tfrecord import extract_frame, WAYMO_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument('split', choices=['training', 'validation'])
parser.add_argument('--out-dir', default='./tmp')
parser.add_argument('--resize', default=1.0, type=float)
args = parser.parse_args()

url_template = 'gs://waymo_open_dataset_v_1_2_0/{split}/{split}_%04d.tar'.format(split=args.split)
if args.split == 'training':
    num_segs = 32
elif args.split == 'validation':
    num_segs = 8

os.makedirs(args.out_dir, exist_ok=True)

clip_id = len(glob.glob('labels/*.txt'))
for seg_id in range(0, num_segs):
    flag = os.system('gsutil cp ' + url_template % seg_id + ' ' + args.out_dir)
    assert flag == 0, 'Failed to download segment %d. Make sure gsutil is installed'%seg_id
    os.system('cd %s; tar xf %s_%04d.tar'%(args.out_dir, args.split, seg_id))
    tfrecords = sorted(glob.glob('%s/*.tfrecord'%args.out_dir))
    for record in tfrecords:
        dir_name = str(clip_id)
        image_dir = os.path.join(args.out_dir, 'images', dir_name)
        label_path = os.path.join(args.out_dir, 'labels', dir_name + '.txt')
        depth_dir = os.path.join(args.out_dir, 'depth', dir_name)
        calib_dir = os.path.join(args.out_dir, 'calib', dir_name)
        points_dir = os.path.join(args.out_dir, 'points', dir_name)
        extract_frame(record, label_path, image_dir, depth_dir, calib_dir, points_dir, resize_ratio=args.resize)
        print("Clip %d done"%clip_id)
        clip_id += 1
        os.remove(record)

    print("Segment %d done"%seg_id)
