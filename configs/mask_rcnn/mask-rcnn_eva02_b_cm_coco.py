# ============================================================
# Mask R-CNN + ViT-EVA02-Base
# COCO dataset (80 classes)
# ============================================================

default_scope = "mmdet"  # Set the default scope to use MMDetection's registry
custom_imports = dict(
    imports=["mmpretrain.models"], allow_failed_imports=False
)  # Import custom modules (here, mmpretrain.models for ViT-EVA02 backbone)

# ------------------------------------------------------------
# Dataset setup
# ------------------------------------------------------------
dataset_type = "CocoDataset"  # Define the dataset type (COCO format)
data_root = "../datasets/coco/"  # Root directory for the COCO dataset

metainfo = dict(  # Meta information: COCO has 80 object classes
    classes=(
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    )
)

train_pipeline = [  # Training data processing pipeline
    dict(type="LoadImageFromFile", backend_args=None),  # Load image from file
    dict(
        type="LoadAnnotations", with_bbox=True, with_mask=True
    ),  # Load bounding box and mask annotations
    dict(
        type="RandomChoiceResize",  # Resize to one of several scales
        scales=[
            (480, 360),
            (512, 384),
            (576, 432),
            (640, 480),
            (704, 528),
            (768, 576),
            (832, 624),
        ],
        keep_ratio=False,
    ),
    dict(
        type="RandomFlip", prob=0.5
    ),  # Randomly flip image horizontally with 50% probability
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=40,
        contrast_range=(0.6, 1.4),
        saturation_range=(0.6, 1.4),
        hue_delta=20,
    ),
    dict(type="PackDetInputs"),  # Format data for model input
]

test_pipeline = [  # Test data processing pipeline
    dict(type="LoadImageFromFile", backend_args=None),  # Load image from file
    dict(type="Resize", scale=(640, 480), keep_ratio=False),  # Resize to 640x480
    dict(
        type="LoadAnnotations", with_bbox=True, with_mask=True
    ),  # Load bounding box and mask annotations
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),  # Format data for model input, keep meta info for evaluation/visualization
]

train_dataloader = dict(
    batch_size=4,  # Number of images per batch
    num_workers=4,  # Number of worker processes for data loading
    persistent_workers=True,  # Keep workers alive between epochs for efficiency
    sampler=dict(type="DefaultSampler", shuffle=True),  # Shuffle data at each epoch
    batch_sampler=dict(
        type="AspectRatioBatchSampler"
    ),  # Group images with similar aspect ratios for batching
    dataset=dict(
        type=dataset_type,  # Use the defined dataset type (COCO format)
        data_root=data_root,  # Root directory for the dataset
        ann_file="annotations/instances_train2017.json",  # Annotation file for training set
        data_prefix=dict(img="train2017/"),  # Image directory prefix for training set
        filter_cfg=dict(
            filter_empty_gt=True, min_size=32
        ),  # Filter out images without ground truth or too small
        metainfo=metainfo,  # Class and palette info
        pipeline=train_pipeline,  # Data processing pipeline for training
    ),
)

val_dataloader = dict(
    batch_size=1,  # Number of images per batch during validation
    num_workers=4,  # Number of worker processes for data loading
    persistent_workers=True,  # Keep workers alive between epochs for efficiency
    drop_last=False,  # Do not drop the last incomplete batch
    sampler=dict(
        type="DefaultSampler", shuffle=False
    ),  # Do not shuffle validation data
    dataset=dict(
        type=dataset_type,  # Use the defined dataset type (COCO format)
        data_root=data_root,  # Root directory for the dataset
        ann_file="annotations/instances_val2017.json",  # Annotation file for validation set
        data_prefix=dict(img="val2017/"),  # Image directory prefix for validation set
        test_mode=True,  # Enable test mode (no ground truth filtering, etc.)
        metainfo=metainfo,  # Class and palette info
        pipeline=test_pipeline,  # Data processing pipeline for validation/testing
    ),
)

test_dataloader = (
    val_dataloader  # Use the same dataloader settings for testing as for validation
)

# ------------------------------------------------------------
# Evaluation metrics
# ------------------------------------------------------------
val_evaluator = dict(
    type="CocoMetric",  # Use COCO metrics for evaluation
    ann_file=data_root
    + "annotations/instances_val2017.json",  # Annotation file for validation set
    metric=["bbox", "segm"],  # Evaluate both bounding box and segmentation mask
    format_only=False,  # Do not only format results, also compute metrics
)
test_evaluator = (
    val_evaluator  # Use the same evaluator settings for testing as for validation
)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
model = dict(
    type="MaskRCNN",  # Use Mask R-CNN architecture
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],  # Mean for normalization (RGB, ImageNet)
        std=[58.395, 57.12, 57.375],  # Std for normalization (ImageNet)
        bgr_to_rgb=True,  # Convert BGR to RGB
        pad_mask=True,  # Pad mask to match image size
        pad_size_divisor=32,  # Pad image/mask to be divisible by patch size
    ),
    # Backbone: Vision Transformer EVA02-Base from mmpretrain
    backbone=dict(
        type="mmpretrain.ViTEVA02",  # Use ViT-EVA02 backbone from mmpretrain
        arch="b",  # Base variant
        patch_size=14,  # Patch size for ViT
        img_size=224,  # Input image size
        sub_ln=True,  # Sub-layer normalization
        with_cls_token=False,  # No class token for detection
        final_norm=True,  # Apply final normalization
        out_type="featmap",  # Output feature map
        out_indices=(11,),  # Use last layer output
        frozen_stages=-1,  # Do not freeze any stages
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-base-p14_pre_in21k_20230505-2f2d4d3c.pth",
            prefix="backbone.",  # Prefix for loading weights
        ),
    ),
    # The ChannelMapper neck adapts the single-scale ViT output to the format expected by the detection heads
    neck=dict(
        type="ChannelMapper",
        in_channels=[768],  # ViT-EVA02-Base output channels
        kernel_size=1,  # 1x1 conv for channel mapping
        out_channels=256,  # Output channels for FPN compatibility
        act_cfg=None,  # No activation
        norm_cfg=dict(type="GN", num_groups=32),  # GroupNorm
        num_outs=1,  # Single output scale
    ),
    # The RPN (Region Proposal Network) head generates object proposals from the feature map
    # It predicts objectness scores and bounding box deltas for anchors at each spatial location
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,  # Match neck output
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[2, 4, 8, 16, 32],  # Multiple scales for better detection
            ratios=[0.5, 1.0, 2.0],  # Anchor aspect ratios
            strides=[14],
        ),  # Match patch size
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
        ),  # RPN classification loss
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),  # RPN bbox regression loss
    ),
    # The RoI (Region of Interest) head performs classification, bounding box regression, and mask prediction for each detected object proposal
    roi_head=dict(
        type="StandardRoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(
                type="RoIAlign", output_size=7, sampling_ratio=2
            ),  # Extract 7x7 RoI features
            out_channels=256,
            featmap_strides=[14],
        ),  # Match stride to patch size
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=256,
            fc_out_channels=2048,
            roi_feat_size=7,
            num_classes=80,  # COCO has 80 classes
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            norm_cfg=dict(type="GN"),  # GroupNorm for stability
            loss_cls=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),  # RoI classification loss
            loss_bbox=dict(
                type="SmoothL1Loss", beta=1.0, loss_weight=1.0
            ),  # RoI bbox regression loss
        ),
        mask_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(
                type="RoIAlign", output_size=14, sampling_ratio=2
            ),  # Extract 14x14 RoI features for mask
            out_channels=256,
            featmap_strides=[14],
        ),  # Match stride to patch size
        mask_head=dict(
            type="FCNMaskHead",
            num_convs=8,  # Number of conv layers in mask head
            in_channels=256,
            conv_out_channels=384,
            num_classes=80,  # COCO has 80 classes
            norm_cfg=dict(type="GN", num_groups=32),  # GroupNorm for stability
            loss_mask=dict(
                type="CrossEntropyLoss", use_mask=True, loss_weight=1.0
            ),  # Mask loss
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,  # IoU threshold for positive anchors
                neg_iou_thr=0.3,  # IoU threshold for negative anchors
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,  # Number of samples per image
                pos_fraction=0.5,  # Fraction of positive samples
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=3000,  # Number of boxes before NMS
            max_per_img=2000,  # Max proposals per image
            nms=dict(type="nms", iou_threshold=0.7),  # NMS for proposals
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,  # IoU threshold for positive RoIs
                neg_iou_thr=0.5,  # IoU threshold for negative RoIs
                min_pos_iou=0.5,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=1024,  # Number of RoIs per image
                pos_fraction=0.33,  # Fraction of positive RoIs
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            mask_size=28,  # Output mask size
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,  # Number of boxes before NMS at test time
            max_per_img=2000,  # Max proposals per image
            nms=dict(type="nms", iou_threshold=0.7),  # NMS for proposals
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.02,  # Score threshold for final detections
            nms=dict(type="nms", iou_threshold=0.5),  # NMS for final detections
            max_per_img=200,  # Max detections per image
            mask_thr_binary=0.5,
        ),  # Threshold for mask binarization
    ),
)

# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------
optim_wrapper = dict(
    type="AmpOptimWrapper",  # Use the standard optimizer wrapper
    optimizer=dict(
        type="AdamW",  # AdamW optimizer (Adam with decoupled weight decay)
        lr=0.0001,  # Lower learning rate for stability (especially for ViT backbones)
        betas=(0.9, 0.999),  # AdamW beta parameters
        weight_decay=0.05,  # Weight decay for regularization
    ),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(
                lr_mult=0.1, decay_mult=1.0
            ),  # Much lower LR for pretrained backbone
            "norm": dict(
                decay_mult=0.0
            ),  # No weight decay for normalization layers (e.g., LayerNorm, GroupNorm)
        }
    ),
    clip_grad=dict(
        max_norm=35, norm_type=2
    ),  # Gradient clipping to avoid exploding gradients
    loss_scale="dynamic",  # Dynamic loss scaling for FP16
)

param_scheduler = [
    dict(
        type="LinearLR",  # Linear learning rate warmup
        start_factor=0.001,  # Start at 0.1% of base LR
        by_epoch=False,  # Warmup by iteration
        begin=0,  # Start from the first iteration
        end=1000,
    ),  # Warmup for 1000 iterations
    dict(
        type="MultiStepLR",  # MultiStep learning rate decay
        begin=0,
        end=36,  # End at 36 epochs
        by_epoch=True,
        milestones=[24, 32],  # Decay at epochs 24 and 32
        gamma=0.1,
    ),  # Decay LR by a factor of 10 at each milestone
]

# ------------------------------------------------------------
# Training, Validation, and Testing Loops
# ------------------------------------------------------------
train_cfg = dict(
    type="EpochBasedTrainLoop",  # Use epoch-based training loop
    max_epochs=36,  # Train for 36 epochs
    val_interval=2,  # Run validation every 2 epochs
)
val_cfg = dict(type="ValLoop")  # Standard validation loop
test_cfg = dict(type="TestLoop")  # Standard test loop

# ------------------------------------------------------------
# Default Hooks
# ------------------------------------------------------------
default_hooks = dict(
    timer=dict(type="IterTimerHook"),  # Measure iteration time
    logger=dict(
        type="LoggerHook", interval=50
    ),  # Log training info every 50 iterations
    param_scheduler=dict(type="ParamSchedulerHook"),  # Update learning rate scheduler
    # early_stopping=dict(
    #     type='EarlyStoppingHook',
    #     monitor='coco/segm_mAP',
    #     patience=10,  # Stop if no improvement for 10 epochs
    #     rule='greater'
    # ),
    checkpoint=dict(
        type="CheckpointHook",
        interval=2,  # Save checkpoint every 2 epochs
        max_keep_ckpts=5,  # Keep the latest 5 checkpoints
        save_best="coco/segm_mAP",  # Save the best checkpoint based on mask mAP
        rule="greater",
    ),
    sampler_seed=dict(
        type="DistSamplerSeedHook"
    ),  # Set random seed for distributed sampler
    visualization=dict(
        type="DetVisualizationHook",
        draw=True,  # Enable visualization
        interval=200,  # Visualize every 200 iterations
    ),
)

# ------------------------------------------------------------
# Visualization and logging
# ------------------------------------------------------------
vis_backends = [
    dict(type="LocalVisBackend"),  # Save visualizations locally
    dict(type="TensorboardVisBackend"),  # Log visualizations to TensorBoard
]
visualizer = dict(
    type="DetLocalVisualizer",  # Use the default detection visualizer
    vis_backends=vis_backends,  # Backends for visualization output
    name="visualizer",  # Visualizer instance name
)

# ------------------------------------------------------------
# Misc settings
# ------------------------------------------------------------
log_level = "INFO"  # Logging level (INFO, WARNING, ERROR, etc.)
log_processor = dict(
    type="LogProcessor",
    window_size=50,  # Number of iterations to average for logging
    by_epoch=True,  # Log by epoch
)
env_cfg = dict(
    cudnn_benchmark=True,  # Enable cuDNN benchmark for faster training on large dataset
    mp_cfg=dict(
        mp_start_method="fork",  # Multiprocessing start method
        opencv_num_threads=0,  # Number of OpenCV threads
    ),
    dist_cfg=dict(backend="nccl"),  # Distributed backend (NCCL for GPUs)
)
work_dir = "./exps"  # Directory to save logs and checkpoints
load_from = None  # Path to load checkpoint for fine-tuning (None to train from scratch)
resume = False  # Whether to resume training from the latest checkpoint
auto_scale_lr = dict(
    enable=False,  # Disable automatic learning rate scaling
    base_batch_size=16,  # Base batch size for LR scaling (if enabled)
)
