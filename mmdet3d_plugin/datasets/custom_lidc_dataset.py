# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import json
import tempfile
from os import path as osp
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.common.data_classes import EvalBoxes
import mmcv
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesMonoDataset, NuScenesDataset, Custom3DDataset
import os
import copy
from mmdet3d.core.bbox import CameraInstance3DBoxes, LiDARInstance3DBoxes, get_box_type, Box3DMode


@DATASETS.register_module()
class CustomLIDCDataset(Custom3DDataset):
    r"""LIDC Dataset.
    This dataset manage LIDC Dataset.
    """

    # replace with all the classes in customized pkl info file
    CLASSES = ('normal', 'nodule')

    def __init__(self,
            ann_file,
            ann_file_2d, 
            pipeline=None,
            data_root=None,
            classes=None,
            load_interval=1,
            with_velocity=False,
            modality=None,
            box_type_3d='Camera',
            filter_empty_gt=False,
            test_mode=False,
            eval_version='detection_cvpr_2019',
            load_separate=False, 
            use_valid_flag=False):        
         
        self.ann_file = ann_file
        self.ann_file_2d = ann_file_2d
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.load_separate = load_separate
        self.with_velocity = with_velocity
        self.eval_version = eval_version

        super(CustomLIDCDataset, self).__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)
        
        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=False,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        self.load_annotations_2d(ann_file_2d)

    def __len__(self):
        return super(CustomLIDCDataset, self).__len__()
    
    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos_ori = data_infos = list(sorted(data['infos'], key=lambda e: e['token']))
        #data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']

        if self.load_separate:
            data_infos_path = []
            out_dir = self.ann_file.split('.')[0]
            for i in mmcv.track_iter_progress(range(len(data_infos_ori))):
                out_file = osp.join(out_dir, '%07d.pkl' % i)
                data_infos_path.append(out_file)
                if not osp.exists(out_file):
                    mmcv.dump(data_infos_ori[i], out_file, file_format='pkl')
            data_infos_path = data_infos_path[::self.load_interval]
            return data_infos_path

        return data_infos
    
    def pre_pipeline(self, results):
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['bbox2d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def load_annotations_2d(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.impath_to_imgid = {}
        self.imgid_to_dataid = {}
        data_infos = []
        total_ann_ids = []
        for i in self.coco.get_img_ids():
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            self.impath_to_imgid['./data/lidc/' + info['file_name']] = i
            #self.impath_to_imgid[info['file_name']] = i
            self.imgid_to_dataid[i] = len(data_infos)
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        self.data_infos_2d = data_infos

    def impath_to_ann2d(self, impath):
        img_id = self.impath_to_imgid[impath]
        data_id = self.imgid_to_dataid[img_id]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self.get_ann_info_2d(self.data_infos_2d[data_id], ann_info)

    def get_data_info(self, index):
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        if not self.load_separate:
            info = self.data_infos[index]
        else:
            info = mmcv.load(self.data_infos[index], file_format='pkl')

        

        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            #pts_filename=info['lidar_path'],
            #sweeps=info['sweeps'],
            #timestamp=info['timestamp'] / 1e6,
        )

        image_paths = []
        #lidar2img_rts = []
        intrinsics = []
        extrinsics = []
        img_timestamp = []
        for cam_type, cam_info in info['cams'].items():
            img_timestamp.append(cam_info['timestamp'] / 1e6)
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            #lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            #lidar2cam_t = cam_info[
            #                  'sensor2lidar_translation'] @ lidar2cam_r.T
            #lidar2cam_rt = np.eye(4)
            #lidar2cam_rt[:3, :3] = lidar2cam_r.T
            #lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            #lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            intrinsics.append(viewpad)
            #extrinsics.append(
            #    lidar2cam_rt)  ###The extrinsics mean the tranformation from lidar to camera. If anyone want to use the extrinsics as sensor to lidar, please use np.linalg.inv(lidar2cam_rt.T) and modify the ResizeCropFlipImage and LoadMultiViewImageFromMultiSweepsFiles.
            #lidar2img_rts.append(lidar2img_rt)
            extrinsics.append(np.eye(4))

        input_dict.update(
            dict(
                img_timestamp=img_timestamp,
                img_filename=image_paths,
                #lidar2img=lidar2img_rts,
                intrinsics=intrinsics,
                extrinsics=extrinsics
            ))

        input_dict['img_info'] = info
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

            gt_bboxes_3d = annos['gt_bboxes_3d']  # 3d boxes from data_info (tensor)
            gt_labels_3d = annos['gt_labels_3d']
            gt_bboxes_2d = []  # per-view 2d bboxes
            gt_bboxes_ignore = []  # per-view 2d bboxes
            gt_bboxes_2d_to_3d = []  # mapping from per-view 2d bboxes to 3d bboxes
            gt_labels_2d = []  # mapping from per-view 2d bboxes to 3d bboxes

            for cam_i in range(len(image_paths)):
                ann_2d = self.impath_to_ann2d(image_paths[cam_i])
                labels_2d = ann_2d['labels']
                bboxes_2d = ann_2d['bboxes_2d']
                bboxes_ignore = ann_2d['gt_bboxes_ignore']
                bboxes_cam = ann_2d['bboxes_cam'] # 3d boxes from 2d annotation
                lidar2cam = extrinsics[cam_i].T

                # centers_lidar = gt_bboxes_3d.gravity_center.numpy()
                # centers_lidar_hom = np.concatenate([centers_lidar, np.ones((len(centers_lidar), 1))], axis=1)
                # centers_cam = (centers_lidar_hom @ lidar2cam.T)[:, :3]
                # match = self.center_match(bboxes_cam, centers_cam)
                #print("Image path:", image_paths[cam_i])
                # print("BBOXES_CAM for {}:".format(info['token']), bboxes_cam)
                # print("CENTERS_CAM for {}:".format(info['token']), centers_cam)
                # print("MATCH for {}:".format(info['token']), match)
                # assert (labels_2d[match > -1] == gt_labels_3d[match[match > -1]]).all()

                #centers_gt = gt_bboxes_3d.gravity_center.numpy()
                #centers_cam = bboxes_cam[:, :3] + bboxes_cam[:, 3:6] / 2
                #centers_pred = np.concatenate([centers_cam, np.ones((len(centers_cam), 1))], axis=1)  # Or directly use provided centers
                #match = self.center_match(centers_cam, centers_gt)
                #assert (labels_2d[match > -1] == gt_labels_3d[match[match > -1]]).all()

                match = np.arange(len(bboxes_2d))


                gt_bboxes_2d.append(bboxes_2d)
                gt_bboxes_2d_to_3d.append(match)
                gt_labels_2d.append(labels_2d)
                gt_bboxes_ignore.append(bboxes_ignore)

            annos['gt_bboxes_2d'] = gt_bboxes_2d
            annos['gt_labels_2d'] = gt_labels_2d
            annos['gt_bboxes_2d_to_3d'] = gt_bboxes_2d_to_3d
            annos['gt_bboxes_ignore'] = gt_bboxes_ignore
        return input_dict
    

    def get_ann_info(self, index):
        """Get annotation info according to the given index.
        WE change to use CameraInstance instead of LiDARInstance

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        if not self.load_separate:
            info = self.data_infos[index]
        else:
            info = mmcv.load(self.data_infos[index], file_format='pkl')
        # # filter out bbox containing no points
        # if self.use_valid_flag:
        #     mask = info['valid_flag']
        # else:
        #     mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes']
        gt_names_3d = info['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # if self.with_velocity:
        #     gt_velocity = info['gt_velocity']
        #     nan_mask = np.isnan(gt_velocity[:, 0])
        #     gt_velocity[nan_mask] = [0.0, 0.0]
        #     gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the lidc box center is [0.0, 0.0, 0.0], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = CameraInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.0, 0.0, 0.0)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results
    

    def get_ann_info_2d(self, img_info_2d, ann_info_2d):
        """Parse bbox annotation.

        Args:
            img_info (list[dict]): Image info.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, labels,
                gt_bboxes_3d, gt_labels_3d, attr_labels, centers2d,
                depths, bboxes_ignore, masks, seg_map
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_bboxes_cam3d = []
        for i, ann in enumerate(ann_info_2d):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info_2d['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info_2d['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
               gt_bboxes.append(bbox)
               gt_labels.append(self.cat2label[ann['category_id']])
               bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(1, -1)
               bbox_cam3d = np.concatenate([bbox_cam3d], axis=-1)
               gt_bboxes_cam3d.append(bbox_cam3d.squeeze())

        gt_bboxes.append(bbox)
        gt_labels.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, 6), dtype=np.float32)

        if gt_bboxes_ignore:
           gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes_cam=gt_bboxes_cam3d,
            bboxes_2d=gt_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            labels=gt_labels, )
        return ann
    
    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox',
                         ):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        # Instantiate your dataset class
        dataset = CustomLIDCDataset(data_root='../data/lidc', test_mode=False)

        # Get data info for a sample index
        data_info = dataset.get_data_info(0)

        # Print the result
        print(data_info)

        # Check for edge cases (e.g., empty annotations)
        for idx in range(len(dataset)):
            info = dataset.get_data_info(idx)
            if info is None:
                print(f"Sample {idx} has no valid annotations.")

        return NotImplementedError

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None,):


        return NotImplementedError
