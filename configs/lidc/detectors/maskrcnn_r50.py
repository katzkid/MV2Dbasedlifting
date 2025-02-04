model = dict(
    base_detector=dict(
        type="TwoStageDetBase",
        init_cfg=dict(
            type="Pretrained",
            checkpoint="./weights/mask_rcnn_r50_fpn_1x_nuim_20201008_195238-e99f5182.pth",
        ),
        backbone=dict(
            type="ResNet",
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type="BN", requires_grad=False),
            norm_eval=True,
            style="pytorch",
            # Adjust downsampling in the first convolutional layer
            deep_stem=True,  # Use a "deep stem" for stronger downsampling
            stem_channels=64,  # Match the original ResNet stem
            strides=(2, 2, 2, 2),  # Downsampling stride for each stage
            dilations=(1, 1, 1, 1),
            init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
        ),
        neck=dict(
            type="FPN",
            in_channels=[256, 512, 1024, 2048],  # Input channels from backbone stages
            out_channels=256,
            num_outs=5,  # Generate 5 output feature maps
            start_level=0,  # Start from the first backbone output
            end_level=3,  # End at the last backbone output
            # Add extra downsampling for the 5th feature map (8x8)
            add_extra_convs="on_output",
        ),
        rpn_head=dict(
            type="RPNHead",
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type="AnchorGenerator",
                scales=[1, 2, 4],
                ratios=[0.5, 1.0, 2.0],
                # strides=[4, 8, 16, 32, 64]),  # Match FPN output strides
                strides=[8, 16, 32, 64, 128]),  # Debug: fix problem of having no positive bbox from FPN.
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
            ),
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
        roi_head=dict(
            type="StandardRoIHead",
            bbox_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
            ),  # Match FPN outputs
            bbox_head=dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="L1Loss", loss_weight=1.0),
            ),
        ),
        # model training and testing settings
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.4,  # Debug: fix problem of having no positive bbox from FPN.
                    neg_iou_thr=0.2,  # Debug: fix problem of having no positive bbox from FPN.
                    min_pos_iou=0.2,  # Debug: fix problem of having no positive bbox from FPN.
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False,
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.4,  
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                mask_size=28,
                pos_weight=-1,
                debug=False,
            ),
        ),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type="nms", iou_threshold=0.5),
                max_per_img=100,
                mask_thr_binary=0.5,
            ),
        ),
    )
)
