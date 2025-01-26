# %% [markdown]
# ## Generate data list info
#
# We follow the format of `nuscenes` dataset
#

import os
from pathlib import Path
from os import path as osp
import pickle
import numpy as np
import mmcv
from PIL import Image
import json
import random


def get_id_cross_reference(root_dir, cross_file):
    """
    Get the cross reference of patient_id between LIDC data and synthetic data
    """
    cross_dict = dict()

    cross_file = f"{root_dir}/{cross_file}"
    with open(cross_file, "r") as file:
        for line in file:
            lidc_ref, data_ref = line.strip().split(",")
            cross_dict[data_ref.strip()] = lidc_ref

    return cross_dict


def get_cam_data(root_dir, images_dir, patient_id):
    """
    Get image path for each camera
    """

    images_dir = f"{root_dir}/{images_dir}/Patient{patient_id:04}"
    images_dir = images_dir[1:]  # test

    info = dict()

    for cam in range(10):
        info_cam = dict()

        cam_name = (f"CAM_{cam:02}").upper()
        info_cam["data_path"] = f"{images_dir}/Image_{cam:02}.png"
        info_cam["type"] = cam_name
        info_cam["sample_data_token"] = f"{patient_id:03}cam{cam:02}"
        info_cam["timestamp"] = 0

        # Get cam intrinsic value
        cam_intrinsic_file = (
            f"{root_dir}/Cams/Patient{patient_id:04}/{patient_id:04}_{cam:02}.txt"
        )
        with open(cam_intrinsic_file, "r") as file:
            for line in file:
                cam_intrinsic = [float(value) for value in line.strip().split(",")[:9]]
        info_cam["cam_intrinsic"] = np.array(cam_intrinsic).reshape((3, 3))

        info[cam_name] = info_cam

    return info


def get_3d_annotation(root_dir, anno3d_dir, patient_id):
    """
    Get 3d annotation from raw .txt file
    Return gt_boxed and gt_label
    For LIDC we only have 1 label
    """

    anno_3d_file = f"{root_dir}/{anno3d_dir}/Patient_{patient_id:04}_bbox3d.txt"

    gt_boxes = []
    with open(anno_3d_file, "r") as file:
        for line in file:

            # Convert the comma-separated values into floats
            # the coordinates in the txt file were stored as X, dx, Y, dy, Z, dz
            # Y, X, Z (as well as the extension)
            # we need to convert it to X, Y, Z, dX, dY, dZ
            row = [float(value) for value in line.strip().split(",")]
            row = np.array(row)[[0, 2, 4, 1, 3, 5]]
            row = np.append(row, 0)  # add yaw value

            gt_boxes.append(row)

    gt_boxes = np.array(gt_boxes)
    gt_names = np.array(["nodule"] * len(gt_boxes), dtype="<U32")

    return gt_boxes, gt_names


def get_2d_annotation(root_dir, images_dir, anno_3d_dir, anno_2d_files):
    """Build 2D annotation data"""

    annotations = []
    images = []
    anno_id = 0

    infos_2d_anno = dict()

    for path in anno_2d_files:

        image = dict()

        # Get info about image
        # multiple annotations for 1 image are recorded seperately
        patient_id = path.parts[-2][-4:]
        cam_id = path.parts[-1].split("_")[1]
        # TODO: check if image file exist
        # file_name = f"{root_dir}/{images_dir}/Patient{patient_id}/Image_{cam_id}.png"
        file_name = f"{images_dir}/Patient{patient_id}/Image_{cam_id}.png"  # test
        image["file_name"] = file_name
        image["id"] = f"{patient_id}cam{cam_id}"

        # Get info about cam intrinsic
        cam_intrinsics = get_cam_data(root_dir, images_dir, int(patient_id))
        image["cam_intrinsic"] = cam_intrinsics[f"CAM_{cam_id}"]
        image["width"] = 1024
        image["height"] = 1024

        images.append(image)

        # Get info about bbox 3D
        # TODO: check if the nb_bbox_2d = nb_bbox_3d
        bbox_3ds, _ = get_3d_annotation(root_dir, anno_3d_dir, int(patient_id))

        # Get info about bbox
        with open(path, "r") as file:
            for idx, line in enumerate(file):
                anno2d = dict()

                row = [value for value in line.strip().split(",")]
                x, y, dx, dy = [float(x) for x in row[:4]]
                category_name = row[4]

                anno2d["file_name"] = file_name
                anno2d["image_id"] = f"{patient_id}cam{cam_id}"
                anno2d["area"] = dx * dy
                anno2d["category_name"] = category_name
                anno2d["category_id"] = 0
                anno2d["bbox"] = [x, y, dx, dy]
                anno2d["iscrowd"] = 0
                anno2d["bbox_cam3d"] = bbox_3ds[idx]
                # TODO: check if we need center2d info - list of 3
                anno2d["center2d"] = [0, 0, 0]
                anno2d["id"] = anno_id

                # Additional information
                # subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, texture, malignancy
                add_info = [
                    "subtlety",
                    "internalStructure",
                    "calcification",
                    "sphericity",
                    "margin",
                    "lobulation",
                    "spiculation",
                    "texture",
                    "malignancy",
                ]
                for i, info_type in enumerate(add_info, start=5):
                    anno2d[info_type] = row[i]

                # Write info to list
                annotations.append(anno2d)
                anno_id += 1

    infos_2d_anno["annotations"] = annotations
    infos_2d_anno["images"] = images

    return infos_2d_anno


def train_test_split(data_infos, metadata, train_split=0.8, seed=42):
    """Split data into train_set and test_set"""

    infos_train = dict()
    infos_val = dict()

    # Shuffle the items randomly
    np.random.seed(42)
    random.shuffle(data_infos)

    # Split data
    train_end = int(train_split * len(data_infos))

    train_items = data_infos[:train_end]
    val_items = data_infos[train_end:]

    # train_set = dict(train_items)
    # val_set = dict(val_items)
    train_perc = len(train_items) / len(data_infos) * 100
    val_perc = len(val_items) / len(data_infos) * 100

    print(f"Length train_set: {len(train_items)} [{train_perc:.2f}%]")
    print(f"Length val_set: {len(val_items)} [{val_perc:.2f}%]")

    # Update infos dict
    metadata_train = metadata.copy()
    metadata_val = metadata.copy()
    metadata_train.update({"validation_set": False})
    metadata_val.update({"validation_set": True})

    infos_train.update({"metadata": metadata_train, "infos": train_items})
    infos_val.update({"metadata": metadata_val, "infos": val_items})

    return infos_train, infos_val


def get_annotation_files(anno_2d_path, patient_list):
    # Get 2D annotation paths for patients
    anno_2d_path = [
        folder
        for folder in anno_2d_path.iterdir()
        if folder.is_dir() and folder.name in patient_list
    ]
    anno_2d_files = [
        file for path in anno_2d_path for file in path.rglob("*.txt") if file.is_file()
    ]
    return sorted(anno_2d_files)


def main():

    # Arguments
    root_path = "data/lidc"
    # Root directory to start walking
    # since we're in sandbox folder.
    # root_dir = f"../{root_path}"
    root_dir = root_path

    info_prefix = "lidc"
    version = "v1.0"
    dataset_name = " lidc"
    out_dir = root_dir
    images_dir = "Images"
    anno_3d_dir = "Labels3d"
    anno_2d_dir = "Labels2d"

    #
    # Folder arguments
    #
    db_info_save_path = osp.join(out_dir, f"{info_prefix}_dbinfos.pkl")
    info_train_path = osp.join(out_dir, f"{info_prefix}_infos_train.pkl")
    info_train_2d_anno_path = osp.join(
        out_dir, f"{info_prefix}_infos_train_2d_anno.coco.json"
    )
    info_val_path = osp.join(out_dir, f"{info_prefix}_infos_val.pkl")
    info_val_2d_anno_path = osp.join(out_dir, f"{info_prefix}_infos_val_2d_anno.coco.json")
    error_log_path = osp.join(out_dir, f"{info_prefix}_error_logs.txt")

    ###################################
    # Build 3D ground truth for train and val. Save in pickle files.
    # follow nuScence dataset.
    ###################################

    # Get the cross reference
    cross_file = "patients_processed.txt"
    cross_ref = get_id_cross_reference(root_dir, cross_file)
    nb_patient = len(cross_ref)

    # Initalize list
    lidc_infos_train = dict()  # Store all data info
    logs = dict()  # Store error logs (if any)

    # Build metainfo of the dataset
    metadata = {
        "categories": {
            "nodule": 0,
        },
        "dataset": "lidc",
        "version": "v1.0",
        "info_version": "1.0",
    }
    # lidc_infos_train['metadata'] = metadata

    # Build the ground truth 3D database
    infos = []  # Store all datalist (inside db_infos)

    for i in range(nb_patient):

        info_data = dict()

        info_data["token"] = i

        try:
            # Build patient_id meta data
            info_data["sample_id"] = i
            info_data["lidc_id_ref"] = cross_ref[str(i)]

            # Build cams data
            info_data["cams"] = get_cam_data(root_dir, images_dir=images_dir, patient_id=i)

            # Build 3D annotation data
            info_data["gt_boxes"], info_data["gt_names"] = get_3d_annotation(
                root_dir, anno_3d_dir, i
            )

            # Set valid flag = True b.c we dont have any lidars_ptd or radars_pts
            info_data["valid_flag"] = np.array([True] * len(info_data["gt_boxes"]))

            # Build 2D annotation paths
            # info_data['cam_instances'] = get_2d_annotation(root_dir, anno_2d_dir, i)

            # Write to datalist
            infos.append(info_data)

        except Exception as e:
            logs[f"Patient_{i:04}"] = str(e)
            continue

    print("Finish creating infos database")

    # Split to Train - Validation dataset.
    lidc_infos_train, lidc_infos_val = train_test_split(
        data_infos=infos, metadata=metadata, train_split=0.8
    )

    # Write to disk
    with open(info_train_path, "wb") as f:
        pickle.dump(lidc_infos_train, f)
        print(f"Write train set info into {info_train_path}")

    with open(info_val_path, "wb") as f:
        pickle.dump(lidc_infos_val, f)
        print(f"Write validation set info into {info_val_path}")

    # Write error log file
    with open(error_log_path, "w") as f:
        for key, value in logs.items():
            f.write(f"{key}, {value}\n")
        print(f"Write error log info into {error_log_path}")

    ###################################
    # Generate `infos_train_2d_anno` for 2D Annotation file with COCO `.json` format
    ###################################

    # 2D annotation path
    anno_2d_dir = f"{root_dir}/Labels2d"
    anno_2d_path = Path(anno_2d_dir)

    # Get idx of train_patient and val_patient
    train_patient = [f"Patient{i['sample_id']:04}" for i in lidc_infos_train["infos"]]
    val_patient = [f"Patient{i['sample_id']:04}" for i in lidc_infos_val["infos"]]

    # Get annotation 2D *.txt files for train and validation patients
    train_anno_2d_files = get_annotation_files(anno_2d_path, train_patient)
    print(
        f"Get annotation 2D *.txt files for train patients [{len(train_anno_2d_files)} files]"
    )

    val_anno_2d_files = get_annotation_files(anno_2d_path, val_patient)
    print(
        f"Get annotation 2D *.txt files for validation patients [{len(val_anno_2d_files)} files]"
    )

    # Initialize data dict
    lidc_infos_train_2d_anno = dict()
    lidc_infos_val_2d_anno = dict()

    # Create label data
    categories_data = [{"id": 0, "name": "nodule"}]

    # Build 2D annotation data
    lidc_infos_train_2d_anno = get_2d_annotation(
        root_dir=root_dir,
        images_dir=images_dir,
        anno_3d_dir=anno_3d_dir,
        anno_2d_files=train_anno_2d_files,
    )
    lidc_infos_train_2d_anno["categories"] = categories_data
    print("Finish building 2d annotation for train dataset")

    lidc_infos_val_2d_anno = get_2d_annotation(
        root_dir=root_dir,
        images_dir=images_dir,
        anno_3d_dir=anno_3d_dir,
        anno_2d_files=val_anno_2d_files,
    )
    lidc_infos_val_2d_anno["categories"] = categories_data
    print("Finish building 2d annotation for val dataset")

    # Write to disk
    class NumpyEncoder(json.JSONEncoder):
        """Handle n.array when writing to JSON"""

        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert ndarray to list
            return super(NumpyEncoder, self).default(obj)

    with open(info_train_2d_anno_path, "w") as f:
        # pickle.dump(lidc_infos_train_2d_anno, f)
        json.dump(lidc_infos_train_2d_anno, f, cls=NumpyEncoder, indent=4)
        print(f"Write 2d annotation for train at {info_train_2d_anno_path}")

    with open(info_val_2d_anno_path, "w") as f:
        # pickle.dump(lidc_infos_train_2d_anno, f)
        json.dump(lidc_infos_val_2d_anno, f, cls=NumpyEncoder, indent=4)
        print(f"Write 2d annotation for validation at {info_val_2d_anno_path}")


if __name__ == "__main__":
    main()
