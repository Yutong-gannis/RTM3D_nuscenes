"""
Script for convert nuScenes (KITTI format) to COCO format
"""

from __future__ import absolute_import, annotations
from __future__ import division
from __future__ import print_function

import _init_paths

import math
import os
import argparse
from cv2 import line
import numpy as np
import json
import cv2
from tqdm import tqdm

from utils.ddd_utils import compute_box_3d, project_to_image
from lyft_dataset_sdk.lyftdataset import LyftDataset


class nuScenes2COCO:
    def __init__(self, data_path, output_dir, nuscenes_path) -> None:
        # directory
        self.data_path = data_path
        self.output_dir = output_dir
        self.nuscenes_path = nuscenes_path
        self.json_path = os.path.join(self.nuScenes_path, 'v1.0-trainval')

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        self.nuscenes = LyftDataset(data_path=self.nuscenes_path, json_path=self.json_path)

        # category (9 items)
        self.categories = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
            'traffic_cone', 'barrier']

        # detection categories (6 items)
        self.det_cats = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
            'traffic_cone', 'barrier']

        self.cat_ids = {cat: i + 1 for i, cat in enumerate(self.categories)}
        self.det_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.cat_info = []
        for i, cat in enumerate(self.det_cats):
            self.cat_info.append({'name': cat, 'id': i+1})

    def nuscenes_to_coco(self, img_shape=[900, 1600, 3]):
        # loop to splits
        self.ann_dir = os.path.join(self.data_path, 'label/')
        self.calib_dir = os.path.join(self.data_path, 'calib/')

        # image shape
        self.img_shape = img_shape
        
        self.splits_imagesets = ['train', 'val']

        # loop thru image_sets
        for split in self.splits_imagesets:
            self.ret = {
                'images': [], # contain image_info
                'annotations': [], # contain annotations data
                'categories': self.cat_info
            }
            self.image_set = open(self.data_path + f'{split}.txt', 'r')

            # line = token_to_write
            for line in tqdm(self.image_set):
                if line[-1] == '\n':
                    line = line[:-1]
                self.image_id = line
                
                # TODO: convert image name to line 

                # image filename
                filename = os.path.join(self.nuscenes_path, 'image', line) + '.png'

                # read calibration file
                calib_path = self.calib_dir + f'{line}.txt'
                self.calib = self.read_calib(calib_path)

                self.image_info = {
                    'file_name': f'{filename}', # nuscenes in jpeg
                    'id': self.image_id,
                    'calib': self.calib.tolist()
                    }
                # add image_info to annotations
                self.ret['images'].append(self.image_info)

                # skip annotations part if split was test
                if split == 'test':
                    continue

                # annotations
                ann_path = self.ann_dir + f'{line}.txt'
                annotation_file = open(ann_path, 'r')
                self.read_annotations(annotation_file)
        
            print('# images:', len(self.ret['images']))
            print('# annotations:', len(self.ret['annotations']))

            # save json
            output_file = self.output_dir + f'/{split}.json'
            json.dump(self.ret, open(output_file, 'w'))

    def read_annotations(self, annotations_file):
        for ann_id, line in enumerate(annotations_file):
            ann = line[:-1].split(' ')
            cat_id = self.cat_ids[ann[0]]
            truncated = int(float(ann[1]))
            occluded = int(ann[2])
            alpha = float(ann[3])
            dim = [float(ann[8]), float(ann[9]), float(ann[10])]
            location = [float(ann[11]), float(ann[12]), float(ann[13])]
            rotation_y = float(ann[14])

            num_keypoints = 0
            box_2d_as_point = [0] * 27
            bbox = [0., 0., 0., 0.]
            calib_list = np.reshape(self.calib, (12)).tolist()
            if cat_id in self.det_ids:
                # image = cv2.imread(os.path.join(self.image_set_path, self.image_info['file_name']))
                bbox = [float(ann[4]), float(ann[5]), float(ann[6]), float(ann[7])]
                box_3d = compute_box_3d(dim, location, rotation_y)
                box_2d_as_point, vis_num, pts_center = project_to_image(box_3d, self.calib, self.img_shape)
                box_2d_as_point = np.reshape(box_2d_as_point, (1, 27))
                box_2d_as_point = box_2d_as_point.tolist()[0]
                num_keypoints = vis_num
            
                alpha = rotation_y - math.atan2(pts_center[0, 0] - self.calib[0, 2], self.calib[0, 0])
                annotation = {
                    'segmentation': [[0,0,0,0,0,0]],
                    'num_keypoints':num_keypoints,
                    'area':1,
                    'iscrowd': 0,
                    'keypoints': box_2d_as_point,
                    'image_id': self.image_id,
                    'bbox': self._bbox_to_coco_bbox(bbox),
                    'category_id': cat_id,
                    'id': int(len(self.ret['annotations']) + 1),
                    'dim': dim,
                    'rotation_y': rotation_y,
                    'alpha': alpha,
                    'location':location,
                    'calib':calib_list
                    }
                self.ret['annotations'].append(annotation)
                
    def read_calib(self, calib_path):
        calib_file = open(calib_path, 'r')
        for i, line in enumerate(calib_file):
            if i == 2: # read P2 only
                calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                calib = calib.reshape(3, 4)
                return calib

    def _bbox_to_coco_bbox(self, bbox):
        return [(bbox[0]), (bbox[1]),
                (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

def main():
    parser = argparse.ArgumentParser(description='KITTI to COCO')
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/RTM3D/kitti_format/data/nuscenes_kitti/', help='nuScenes data path')
    parser.add_argument('--nuscenes_path', type=str, default='/root/autodl-tmp/RTM3D/kitti_format/data/nuscenes/', help='nuScenes data path')
    parser.add_argument('--output_path', type=str, default='/root/autodl-tmp/RTM3D/kitti_format/data/nuscenes_kitti/annotations/', help='JSON output path')
    parser.add_argument('--image_shape', type=int, nargs='+', default=[900, 1600, 3], help='Image shape [h, w, c]')
    args = parser.parse_args()

    converter = nuScenes2COCO(data_path=args.data_path, output_dir=args.output_path, nuscenes_path=args.nuscenes_path)
    converter.nuscenes_to_coco(img_shape=args.image_shape)

if __name__ == '__main__':
    main()
    main()
