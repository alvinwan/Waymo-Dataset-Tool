# tensorflow2_latest_p37
import tensorflow as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2
import os
import argparse

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

try:
    tf.enable_eager_execution()
except Exception as e:
    print(e)
WAYMO_CLASSES = ['TYPE_UNKNOWN', 'TYPE_VECHICLE', 'TYPE_PEDESTRIAN', 'TYPE_SIGN', 'TYPE_CYCLIST']

def extract_frame(frames_path, outname, outdir_img, outdir_depth, outdir_calib, outdir_points, class_mapping=WAYMO_CLASSES, resize_ratio=1.0):

    dataset = tf.data.TFRecordDataset(frames_path, compression_type='')
    id_dict = {}
    bboxes_all = {}
    scores_all = {}
    cls_inds_all = {}
    track_ids_all = {}
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    os.makedirs(outdir_img, exist_ok=True)
    os.makedirs(outdir_depth, exist_ok=True)
    os.makedirs(outdir_calib, exist_ok=True)

    for fidx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        (range_images, camera_projections, range_image_top_pose) = (
            frame_utils.parse_range_image_and_camera_projection(frame))
        time = frame.context.stats.time_of_day
        weather = frame.context.stats.weather

        # Pack object bounding boxes
        labels = frame.camera_labels
        if len(labels) == 0:
            labels = frame.projected_lidar_labels
        if len(labels) == 0:
            break
        assert labels[0].name == 1
        boxes, types, ids = extract_labels(labels[0])
        bboxes, cls_inds, track_ids = convert_kitti(boxes, types, ids, id_dict)
        bboxes *= resize_ratio
        scores = np.zeros(cls_inds.shape, dtype='f')
        bboxes_all[fidx] = bboxes
        scores_all[fidx] = scores
        cls_inds_all[fidx] = cls_inds
        track_ids_all[fidx] = track_ids

        # Write image
        im = tf.image.decode_jpeg(frame.images[0].image).numpy()[:,:,::-1]
        target_size = (int(im.shape[1] * resize_ratio), int(im.shape[0] * resize_ratio))
        im = cv2.resize(im, target_size)
        cv2.imwrite(outdir_img + '/%04d.png'%fidx, im)

        # write point cloud
        points_all, cp_points_all, images = writepoints(outdir_points + '/%04d.npy'%fidx, frame, range_images, camera_projections, range_image_top_pose)

        # write depth maps
        writedepth(outdir_depth + '/%04d.png'%fidx, frame, points_all, cp_points_all, images)

        # write calib
        writecalib(outdir_calib + '/%04d.txt'%fidx, frame)

    if len(bboxes_all) > 0:
        writeKITTI(outname, bboxes_all, scores_all, cls_inds_all, track_ids_all, class_mapping)

def extract_labels(camera_label):
    box_labels = camera_label.labels
    boxes = []
    types = []
    ids = []
    for box_label in box_labels:
        boxes.append([box_label.box.center_x, box_label.box.center_y, box_label.box.length, box_label.box.width])
        types.append(box_label.type)
        ids.append(box_label.id)
    return boxes, types, ids

def convert_kitti(boxes, types, ids, id_dict):
    max_id = max(id_dict.values()) + 1 if len(id_dict) > 0 else 0
    boxes = np.array(boxes)
    if len(boxes) > 0:
        bboxes = np.zeros_like(boxes)
        bboxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
        bboxes[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2
    else:
        bboxes = np.zeros((0,4), dtype='f')
    
    cls_inds = []
    track_ids = []
    for cls, old_id in zip(types, ids):
        if old_id in id_dict:
            track_id = id_dict[old_id]
        else:
            id_dict[old_id] = max_id
            track_id = max_id
            max_id += 1
        cls_inds.append(cls)
        track_ids.append(track_id)
    cls_inds = np.array(cls_inds)
    track_ids = np.array(track_ids)
    return bboxes, cls_inds, track_ids

def writeKITTI(filename, bboxes, scores, cls_inds, track_ids=None, classes=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'w')
    for fid in bboxes:
        for bid in range(len(bboxes[fid])):
            fields = [''] * 17
            fields[0] = fid
            fields[1] = -1 if track_ids is None else int(track_ids[fid][bid])
            fields[2] = classes[int(cls_inds[fid][bid])]
            fields[3:6] = [-1] * 3
            fields[6:10] = bboxes[fid][bid]
            fields[10:16] = [-1] * 6
            fields[16] = scores[fid][bid]
            fields = map(str, fields)
            f.write(' '.join(fields) + '\n')
    f.close()

def writepoints(filename, frame, range_images, camera_projections, range_image_top_pose):
    # projection code taken from https://colab.research.google.com/github/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1)

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # camera projection corresponding to each point.
    cp_points_all = np.concatenate(cp_points, axis=0)
    cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

    images = sorted(frame.images, key=lambda i:i.name)
    cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
    cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

    np.save(filename, points_all)

    return points_all, cp_points_all, images


def writedepth(filename, frame, points_all, cp_points_all, images):
    
    # The distance between lidar points and vehicle frame origin.
    points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

    mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)

    cp_points_all_tensor = tf.cast(tf.gather_nd(
        cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
    points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

    projected_points_all_from_raw_data = tf.concat(
        [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()

    x, y, _ = projected_points_all_from_raw_data.astype(np.uint16).T
    h, w, _ = tf.image.decode_jpeg(frame.images[0].image).numpy()[:,:,::-1].shape
    assert y.max() < h and x.max() < w, (y.max(), h, x.max(), w)

    depth_map = np.zeros((h, w), dtype=np.float32)
    depth_map[y, x] = projected_points_all_from_raw_data[:, 2]
    depth_map = (depth_map * 256.).astype(np.uint16)
    cv2.imwrite(filename, depth_map, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def writecalib(filename, frame):
    with open(filename, 'w') as fh:
        fh.write('camera_id,R00,R01,R02,T0,R10,R11,R12,T1,R20,R21,R22,T2,P30,P31,P32,P33\n')
        for i, camera in enumerate(frame.context.camera_calibrations):
            fh.write(f"{i},{','.join(map(str, camera.extrinsic.transform))}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('record_path')
    parser.add_argument('output_id')
    parser.add_argument('--workdir', default='.')
    parser.add_argument('--resize', default=1.0, type=float)
    args = parser.parse_args()
    os.chdir(args.workdir)
    image_dir = os.path.join('images', args.output_id)
    label_path = os.path.join('labels', args.output_id + '.txt')
    depth_dir = os.path.join('depth', args.output_id)
    calib_dir = os.path.join('calib', args.output_id)
    points_dir = os.path.join('points', args.output_id)
    extract_frame(args.record_path, label_path, image_dir, depth_dir, calib_dir, points_dir, resize_ratio=args.resize)

if __name__ == "__main__":
    main()
