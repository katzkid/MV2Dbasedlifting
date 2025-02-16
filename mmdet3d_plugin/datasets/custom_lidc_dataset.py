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
from mmdet3d.core.bbox import CameraInstance3DBoxes, LiDARInstance3DBoxes, get_box_type, Box3DMode, Coord3DMode
from mmdet3d.core import show_result
import torch


from sandbox.extrinsic_params_calculator import compute_extrinsics, extrinsic_to_homogeneous, world_to_camera_frame #extrinsic params computation


@DATASETS.register_module()
class CustomLIDCDataset(Custom3DDataset):
    r"""LIDC Dataset.
    This dataset manage LIDC Dataset.
    """

    # replace with all the classes in customized pkl info file
    CLASSES = ('nodule',) 

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
        self.data_infos = self.load_annotations(self.ann_file)

    def __len__(self):
        return super(CustomLIDCDataset, self).__len__()

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos_ori = data_infos = list(sorted(data['infos'], key=lambda e: e['token']))
        # data_infos = data_infos[::self.load_interval]
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
            # self.impath_to_imgid[info['file_name']] = i
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
        lidar2img_rts = []
        intrinsics = []
        extrinsics = []
        ###################################################################################
        # Compute extrinsics
        num_cameras = 10
        angle_step = 360 / num_cameras  # Angle separation between cameras
        radius = 300  # Distance from the origin

        # Compute extrinsics for all cameras
        extrinsics_local = []
        for i in range(num_cameras):
            theta = i * angle_step
            R, t = compute_extrinsics(theta, radius)
            extrinsics_local.append((R, t))

        # Display the extrinsics
        # extrinsics

        # Compute homogeneous extrinsic matrices for all cameras
        homogeneous_extrinsics = [extrinsic_to_homogeneous(R, t) for R, t in extrinsics_local]
        extrinsics = homogeneous_extrinsics #add homogeous extrinsic values
        ###################################################################################
        img_timestamp = []
        for cam_type, cam_info in info['cams'].items():
            cam_idx = int(cam_type.split('_')[-1])#added index for cameras
            img_timestamp.append(cam_info['timestamp'] / 1e6)
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            # lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            # lidar2cam_t = cam_info[
            #                  'sensor2lidar_translation'] @ lidar2cam_r.T
            # lidar2cam_rt = np.eye(4)
            # lidar2cam_rt[:3, :3] = lidar2cam_r.T
            # lidar2cam_rt[3, :3] = -lidar2cam_t
            lidar2cam_rt = extrinsics[cam_idx]
            intrinsic = cam_info['cam_intrinsic']
            #convert intrinsic values from mm to pixels
            pixel_size = 800/1024
            intrinsic[0, 0] = intrinsic[0, 0] / pixel_size
            intrinsic[1, 1] = intrinsic[1, 1] / pixel_size
            intrinsic[0, 2] = intrinsic[0, 2] / pixel_size
            intrinsic[1, 2] = intrinsic[1, 2] / pixel_size 
            #recenter back to 0,0
            intrinsic[0, 2] = -intrinsic[0, 2]
            intrinsic[1, 2] = -intrinsic[1, 2]

            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            intrinsics.append(viewpad)
            # extrinsics.append(
            #    lidar2cam_rt)  ###The extrinsics mean the tranformation from lidar to camera. If anyone want to use the extrinsics as sensor to lidar, please use np.linalg.inv(lidar2cam_rt.T) and modify the ResizeCropFlipImage and LoadMultiViewImageFromMultiSweepsFiles.
            lidar2img_rts.append(lidar2img_rt)
            # extrinsics.append(np.eye(4))

        # let Extrinsics be lidar2img

        input_dict.update(
            dict(
                img_timestamp=img_timestamp,
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
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

                # debug remove last entry here
                bboxes_2d = bboxes_2d[:-1]
                labels_2d = labels_2d[:-1]

                bboxes_ignore = ann_2d['gt_bboxes_ignore']
                bboxes_cam = ann_2d['bboxes_cam'] # 3d boxes from 2d annotation
                # lidar2cam = extrinsics[cam_i]

                # centers_lidar = gt_bboxes_3d.gravity_center.numpy()
                # centers_lidar_hom = np.concatenate([centers_lidar, np.ones((len(centers_lidar), 1))], axis=1)
                # centers_cam = (centers_lidar_hom @ lidar2cam.T)[:, :3]
                # match = self.center_match(bboxes_cam, centers_cam)
                # print("Image path:", image_paths[cam_i])
                # print("BBOXES_CAM for {}:".format(info['token']), bboxes_cam)
                # print("CENTERS_CAM for {}:".format(info['token']), centers_cam)
                # print("MATCH for {}:".format(info['token']), match)
                # assert (labels_2d[match > -1] == gt_labels_3d[match[match > -1]]).all()

                # centers_gt = gt_bboxes_3d.gravity_center.numpy()
                # centers_cam = bboxes_cam[:, :3] + bboxes_cam[:, 3:6] / 2
                # centers_pred = np.concatenate([centers_cam, np.ones((len(centers_cam), 1))], axis=1)  # Or directly use provided centers
                # match = self.center_match(centers_cam, centers_gt)
                # assert (labels_2d[match > -1] == gt_labels_3d[match[match > -1]]).all()

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
        ###################################################################################
        # Compute extrinsics
        num_cameras = 10
        angle_step = 360 / num_cameras  # Angle separation between cameras
        radius = 300  # Distance from the origin

        # Compute extrinsics for all cameras
        extrinsics_local = []
        for i in range(num_cameras):
            theta = i * angle_step
            R, t = compute_extrinsics(theta, radius)
            extrinsics_local.append((R, t))

        # Display the extrinsics
        # extrinsics

        # Compute homogeneous extrinsic matrices for all cameras
        homogeneous_extrinsics = [extrinsic_to_homogeneous(R, t) for R, t in extrinsics_local]
        ###################################################################################
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
        #print("gt_bboxes_3d", gt_bboxes_3d)
        # Extract the first three coordinates of each bounding box
        gt_bboxes_3d_coords = np.array([bbox[:3] for bbox in gt_bboxes_3d])

        # Convert to camera coordinates
        gt_bboxes_3d_coords_cam = world_to_camera_frame(gt_bboxes_3d_coords, homogeneous_extrinsics[:1])

        #print("gt_bboxes_3d_coords_cam", gt_bboxes_3d_coords_cam)

        # Update each bounding box's coordinates with the new camera-space values
        Xc = []
        for i in range(len(gt_bboxes_3d)):
            gt_bboxes_3d[i][:3] = gt_bboxes_3d_coords_cam[i][0]
            first_element = gt_bboxes_3d_coords_cam[i][0][0]  # Access the first element of the array
            Xc.append(first_element)  # Append to Xc

        #print("gt_bboxes_3d after compute", gt_bboxes_3d)






        # # Combine the transformed cam coordinates with the original dimensions and orientation
        # gt_bboxes_3d = torch.cat([
        #     gt_bboxes_3d_coords_cam,  # Transformed [x, y, z]
        #     gt_bboxes_3d[:, 3:6],  # Original [l, w, h]
        #     gt_bboxes_3d[:, 6:7],  # Original yaw (may need adjustment)
        # ], dim=1)
        ################################################################################################

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
        
        #Scale down the gt_bboxes_3d by scaling_factor
        #gt_bboxes_3d = scale_camera_boxes(gt_bboxes_3d, scaling_factor)

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
        ###################################################################################
        num_cameras = 10
        angle_step = 360 / num_cameras  # Angle separation between cameras
        radius = 300  # Distance from the origin

        # Compute extrinsics for all cameras
        extrinsics = []
        for i in range(num_cameras):
            theta = i * angle_step
            R, t = compute_extrinsics(theta, radius)
            extrinsics.append((R, t))

        # Compute homogeneous extrinsic matrices for all cameras
        homogeneous_extrinsics = [extrinsic_to_homogeneous(R, t) for R, t in extrinsics]
        ###################################################################################
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_bboxes_cam3d = []
        for i, ann in enumerate(ann_info_2d):
            if ann.get('ignore', False):
                continue
            y1, x1, h, w = ann["bbox"]  # switch order of x1, y1
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
                # print("bbox_cam3d_world", bbox_cam3d[0][:3])
                # convert to camera instance 3d boxes
                bbox_cam3d[0][:3] = np.array(world_to_camera_frame(np.array([bbox_cam3d[0][:3]]), homogeneous_extrinsics[:1])[0])
                #Adjust for the SID (Source is at 300,0,0 and detector is at -300,0,0)
                SID = 600
                bbox_cam3d[0][0] += SID
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

    #
    #  Codes used for validation and inference.
    #

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det, self.with_velocity)
            sample_token = self.data_infos[sample_id]["token"]
            # boxes = lidar_nusc_box_to_global(
            #     self.data_infos[sample_id],
            #     boxes,
            #     mapped_class_names,
            #     self.eval_detection_configs,
            #     self.eval_version,
            # )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                # if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                #     if name in [
                #         "car",
                #         "construction_vehicle",
                #         "bus",
                #         "truck",
                #         "trailer",
                #     ]:
                #         attr = "vehicle.moving"
                #     elif name in ["bicycle", "motorcycle"]:
                #         attr = "cycle.with_rider"
                #     else:
                #         attr = NuScenesDataset.DefaultAttribute[name]
                # else:
                #     if name in ["pedestrian"]:
                #         attr = "pedestrian.standing"
                #     elif name in ["bus"]:
                #         attr = "vehicle.stopped"
                #     else:
                #         attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=None,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_lidc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                if name in ['pts_bbox', 'img_bbox']:
                    print(f'\nFormating bboxes of {name}')
                    results_ = [out[name] for out in results]
                    tmp_file_ = osp.join(jsonfile_prefix, name)
                    result_files.update(
                        {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

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

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None and not isinstance(tmp_dir, str):
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict

    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, "Expect out_dir, got none."
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if "pts_bbox" in result.keys():
                result = result["pts_bbox"]
            data_info = self.data_infos[i]
            pts_path = data_info["lidar_path"]
            file_name = osp.split(pts_path)[-1].split(".")[0]
            points = self._extract_data(i, pipeline, "points").numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(
                points, Coord3DMode.LIDAR, Coord3DMode.DEPTH
            )
            inds = result["scores_3d"] > 0.1
            gt_bboxes = self.get_ann_info(i)["gt_bboxes_3d"].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(
                gt_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH
            )
            pred_bboxes = result["boxes_3d"][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(
                pred_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH
            )
            show_result(
                points, show_gt_bboxes, show_pred_bboxes, out_dir, file_name, show
            )

    def scale_camera_boxes(self, boxes, scaling_factor):
        """
        Scale down 3D bounding boxes in a `CameraInstance3DBoxes` object by given scaling factors.
        Each element in the scaling_factor list corresponds to a specific box.

        Args:
            boxes (CameraInstance3DBoxes): The input box object to be scaled.
            scaling_factor (float | list[float]): The scaling factors to reduce the size of the boxes.
                If a single float is provided, it will be applied to all boxes.
                Must be between 0 and 1.

        Returns:
            CameraInstance3DBoxes: A new `CameraInstance3DBoxes` object with scaled-down boxes.
        """
        # Extract the current tensor representation
        tensor = boxes.tensor

        if tensor.numel() == 0:
            return CameraInstance3DBoxes(tensor, box_dim=tensor.shape[1], with_yaw=boxes.with_yaw)

        # Get center coordinates (gravity_center)
        centers = boxes.gravity_center  # Shape: (N, 3)
        print("centers", centers)
        
        # Compute the original extents
        original_extents = tensor[:, [3, 4, 5]]  # dx, dy, dz

        # Ensure scaling_factor is a list and has the same length as the number of boxes
        if isinstance(scaling_factor, float):
            scaling_factor = [scaling_factor] * len(tensor)
        else:
            assert len(scaling_factor) == len(tensor), (
                "The length of scaling_factor must match the number of boxes."
            )

        # Compute the bottom center before scaling
        bottom_centers = boxes.bottom_center
        print("bottom_centers", bottom_centers)

        # Scale the dimensions individually for each box
        scaled_extents = torch.zeros_like(original_extents)
        for i in range(len(tensor)):
            sf = scaling_factor[i]
            scaled_extents[i] = original_extents[i] * sf

        print("scaled_extents", scaled_extents)

        # Reconstruct the new tensor using the bottom center as a reference
        new_tensor = torch.zeros_like(tensor)
        new_centers = centers.copy()
        
        # Adjust y-coordinate based on scaled dy
        new_centers[:, 1] += scaled_extents[:, 1]
        
        print("new_centers", new_centers)

        # Update the new tensor with scaled extents and adjusted centers
        new_tensor[:, :3] = new_centers
        new_tensor[:, 3] = (scaled_extents[:, 0])  # Scaled dx
        new_tensor[:, 4] = (scaled_extents[:, 1]) # Scaled dy
        new_tensor[:, 5] = (scaled_extents[:, 2])  # Scaled dz
        new_tensor[:, 6] = tensor[:, 6]  # Yaw remains the same

        # Create a new CameraInstance3DBoxes object with the scaled tensor
        return CameraInstance3DBoxes(new_tensor, box_dim=tensor.shape[1], with_yaw=boxes.with_yaw)


def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # # our LiDAR coordinate system -> nuScenes box coordinate system
    # nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (0, 0, 0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            # nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list

def lidar_nusc_box_to_global(
    info, boxes, classes, eval_configs, eval_version="detection_cvpr_2019"
):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list





