dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=64)
fp16 = dict(loss_scale=512.0)
img_scale = (1024, 1024)
work_dir ='./work_dirs_wildfire/yolox_swintContrast_SepInd10#OHEM190_ftFIgLib2m/'

model = dict(
    type='YOLOX',
    input_size=(640, 640),
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(
        type='SwinTransformerContrast',
        in_channels = 6,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
        )),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[192, 384, 768],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHeadSepSample',
        num_classes=2,
        in_channels=128,
        feat_channels=128,
        rate=200,
        pos_sample=dict(type='Random', rate=10),
        neg_sample=dict(type='OHEM', rate=190, mu=-300000, mse=500000),
        stacked_convs=2,
        strides=[8, 16, 32],
        use_depthwise=False,
        dcn_on_last_conv=False,
        conv_bias='auto',
        conv_cfg=None,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
dataset_type = 'FireSmokeDatasetFIgLibMultiFrames'
data_root = 'F:/FIgLib/HPWREN-FIgLib-Data/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageListFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='DefaultFormatBundle'),
    dict(type='ImageToTensor', keys=['img_000']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels',"img_000"]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageListFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='ImageToTensor', keys=['img','img_000']),
            dict(type='Collect', keys=['img',"img_000"],
                 meta_keys=('filename', 'ori_filename', 'ori_shape','img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction')),
        ])
]

data = dict(
    # samples_per_gpu=32,
    # workers_per_gpu=24, ## For Train
    samples_per_gpu=2,
    workers_per_gpu=1,  ## For Debug
    train=dict(
        type=dataset_type,
        minutes=[-2],
        ann_file=[data_root + "train/FIgLib_train.txt"],
        img_prefix=[data_root + "train/"],
        #ann_file=[data_root + "test/FlgLib_test_small.txt"],
        #img_prefix=[data_root + "test/"],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        minutes=[-2],
        ann_file= data_root + "test/FIgLib_test.txt",
        img_prefix=data_root + "test/",
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        minutes=[-2],
        ann_file= data_root + "test/FIgLib_test.txt",
        img_prefix=data_root + "test/",
        pipeline=test_pipeline)
)

optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=2,
    num_last_epochs=1,
    min_lr_ratio=0.05)
    
num_last_epochs = 15
runner = dict(type='EpochBasedRunner', max_epochs=80)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

score_thr = 0.05
evaluation = dict(
    save_best='auto',
    interval=1,
    dynamic_intervals=[(25, 1)],
    metric=[''],
    score_thr = score_thr,
    remove1video = False,
    out_file=work_dir+"res.txt"
)
auto_resume = False
gpu_ids = [0]
