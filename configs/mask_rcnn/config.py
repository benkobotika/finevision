_base_ = './mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'

custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)

pretrained = 'https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-tiny-p14_pre_in21k_20230505-d703e7b1.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ViTEVA02',
        arch='t',
        img_size=224,
        patch_size=14,
        out_indices=(3,6,9,11),
        out_type='featmap',
        with_cls_token=False,
        final_norm=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[192, 192, 192, 192],
        out_channels=256,
        num_outs=5
    )
)
