# LIDC

conda env create -f MV2Denv.yml

mkdir weights

download mask_rcnn_r50_fpn_1x_nuim_20201008_195238-e99f5182.pth in weights

mkdir data

cd data

mkdir nuscenes

cd nuscenes
cd ../../

## Setup linux packages
sudo apt update
sudo apt install -y libgl1-mesa-glx

## Sync important folders to S3
MinIO documentation: [client API](https://min.io/docs/minio/linux/reference/minio-mc/mc-mirror.html)
Add `--watch` flag to enter monitoring mode. 

mc mirror --overwrite ./work_dirs s3/kdang/safran/MV2Dbasedlifting/work_dirs
mc mirror --overwrite ./data s3/kdang/safran/MV2Dbasedlifting/data
mc mirror --overwrite ./weights s3/kdang/safran/MV2Dbasedlifting/weights

## Copy from S3 to local
mc cp --recursive s3/kdang/safran/MV2Dbasedlifting/work_dirs .
mc cp --recursive s3/kdang/safran/MV2Dbasedlifting/data .
mc cp --recursive s3/kdang/safran/MV2Dbasedlifting/weights . 



## Train script
bash tools/dist_train.sh configs/lidc/model/mv2d_r50_frcnn_single_frame_roi_1024x1024_ep24_lidc.py 1

### Train 2D Detection model only
bash tools/dist_train.sh configs/lidc/model/mv2d_r50_frcnn_2ddet__toy_ep50.py 1

## Test and inference 
- Run bash script: 

Add `--show` or `--show_dir eval_result/vis` to generate result visualization. 

bash tools/dist_test.sh configs/lidc/model/mv2d_r50_frcnn_single_frame_roi_1024x1024_ep24_lidc.py work_dirs/mv2d_r50_frcnn_single_frame_roi_1024x1024_ep24_lidc/latest.pth 1 --eval bbox --out eval_result/output250127.pkl 

- Run python script
python tools/test.py configs/lidc/model/mv2d_r50_frcnn_single_frame_roi_1024x1024_ep24_lidc.py work_dirs/mv2d_r50_frcnn_single_frame_roi_1024x1024_ep24_lidc/latest.pth 1 --eval bbox --out eval_result/output250126.pkl 




# MV2D
# !wget https://www.nuscenes.org/data/v1.0-mini.tgz  # Download the nuScenes mini split.

# !tar -xf v1.0-mini.tgz -C /data/sets/nuscenes  # Uncompress the nuScenes mini split.

# !pip install nuscenes-devkit &> /dev/null  # Install nuScenes.

Create data nuscenes: python -m tools.create_data nuscenes --root-path data/nuscenes --out-dir data/nuscenes --extra-tag nuscenes --version v1.0-mini

train: bash tools/dist_train.sh configs/mv2d/exp/mv2d_r50_frcnn_single_frame_roi_1408x512_ep24.py 1

This repo is the official PyTorch implementation for paper:   
[Object as Query: Lifting any 2D Object Detector to 3D Detection](https://arxiv.org/abs/2301.02364). Accepted by ICCV 2023.

We design Multi-View 2D Objects guided 3D Object Detector (MV2D), which can lift any 2D object detector to multi-view 3D object detection. Since 2D detections can provide valuable priors for object existence, MV2D exploits 2D detectors to generate object queries conditioned on the rich image semantics. These dynamically generated queries help MV2D to recall objects in the field of view and show a strong capability of localizing 3D objects. For the generated queries, we design a sparse cross attention module to force them to focus on the features of specific objects, which suppresses interference from noises. 

## Preparation
This implementation is built upon [PETR](https://github.com/megvii-research/PETR/tree/main), and can be constructed as the [install.md](https://github.com/megvii-research/PETR/blob/main/install.md).

* Environments  
  Linux, Python == 3.8.10, CUDA == 11.3, pytorch == 1.11.0, mmcv == 1.6.1, mmdet == 2.25.1, mmdet3d == 1.0.0, mmsegmentation == 0.28.0   

* Detection Data   
Follow the mmdet3d to process the nuScenes dataset (https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/data_preparation.md).

* Pretrained weights   
We use nuImages pretrained weights from [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/nuimages). Download the pretrained weights and put them into `weights/` directory. 

* After preparation, you will be able to see the following directory structure:  
  ```
  MV2D
  ├── mmdetection3d
  ├── configs
  ├── mmdet3d_plugin
  ├── tools
  ├── data
  │   ├── nuscenes
  │     ├── ...
  ├── weights
  ├── README.md
  ```

## Train & Inference
<!-- ```bash
git clone https://github.com/tusen-ai/MV2D.git
``` -->
```bash
cd MV2D
```
You can train the model following:
```bash
bash tools/dist_train.sh configs/mv2d/exp/mv2d_r50_frcnn_two_frames_1408x512_ep24.py 1 
```
You can evaluate the model following:
```bash
bash tools/dist_test.sh configs/mv2d/exp/mv2d_r50_frcnn_two_frames_1408x512_ep24.py work_dirs/mv2d_r50_frcnn_two_frames_1408x512_ep24/latest.pth 8 --eval bbox
```

## Main Results
|                                             config                                              |  mAP  |  NDS  |  checkpoint  |
|:-----------------------------------------------------------------------------------------------:|:-----:|:-----:|:------------:|
|    [MV2D-T_R50_1408x512_ep72](./configs/mv2d/exp/mv2d_r50_frcnn_two_frames_1408x512_ep72.py)    | 0.453 | 0.543 | [download](https://drive.google.com/file/d/10zwn2UWb2IzIWqJK1a2y466ZSWoLkD-e/view?usp=drive_link) |  
| [MV2D-S_R50_1408x512_ep72](./configs/mv2d/exp/mv2d_r50_frcnn_single_frame_roi_1408x512_ep72.py) | 0.398 | 0.470 | [download](https://drive.google.com/file/d/139Lsn-UY78ukfOkVlPh6ywnoY_TvdwZn/view?usp=drive_link) |  


## Acknowledgement
Many thanks to the authors of [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) and [petr](https://github.com/megvii-research/PETR/tree/main).

# Citation
If you find this repo useful for your research, please cite
```
@article{wang2023object,
  title={Object as query: Equipping any 2d object detector with 3d detection ability},
  author={Wang, Zitian and Huang, Zehao and Fu, Jiahui and Wang, Naiyan and Liu, Si},
  journal={arXiv preprint arXiv:2301.02364},
  year={2023}
}
```
# Contact
For questions about our paper or code, please contact **Zitian Wang**(wangzt.kghl@gmail.com).
