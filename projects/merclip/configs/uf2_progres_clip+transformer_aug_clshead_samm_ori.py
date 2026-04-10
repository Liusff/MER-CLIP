custom_imports = dict(imports='models')

##backbone drop 0., cls drop 0.2, clip loss ratio 2, fine using coarse

num_frames = 16  #########8
clip_output_dim = 512
cls_output_dim = 512
model = dict(
    type='ActionClip_MA_WithCls_ME',
    vision_encoder_config=dict(
        type='UniFormerV2_clip_extramlp',
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        t_size=num_frames,
        dw_reduction=1.5,
        backbone_drop_path_rate=0.2,
        temporal_downsample=True,
        no_lmhra=False,
        double_lmhra=True,
        return_list=[4, 5, 6, 7, 8, 9, 10, 11],
        n_layers=8,
        n_dim=768,
        n_head=12,
        mlp_factor=4.,
        drop_path_rate=0.4,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        clip_pretrained=False,
        init_cfg=dict(
            checkpoint=
            'work_dirs/uf2_progres_clip+transformer_aug_clshead_dfme_ccac/20240815_090407/best_acc_UF1_epoch_48.pth',
            prefix='vision_encoder.',
            type='Pretrained'),
        output_dim=clip_output_dim,
        cls_output_dim=cls_output_dim,
        clip_drop_out=0.),
    text_encoder_config=dict(
        type='vit_b16',
        output_dim=clip_output_dim,
        context_length=77),
    freeze_text=True,
    freeze_textproj=True,
    to_float32=True,
    labels_or_label_file='data/samm/AUlabel_map.txt',
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        mix=0,
        alpha=0.5,
        mix_prob=0.5,
        format_shape='NCTHW'),
    cls_head_config=dict(
        type='TransfClsHead',
        in_channels=cls_output_dim,
        dropout_ratio=0.1,
        num_classes=5,
        drop_path = 0.0,
        depth = 2,
        num_heads = 4,
        attn_drop_out = 0.0,
        ffn_drop_out = 0.0,
        mlp_mult = 4,
        multi_layer_forward = True,
        average_clips='prob',
        loss_cls=dict(type='FocalLoss', gamma=2)),
    clip_loss_ratio=2.0,
    total_weight=2.0,
    progressive=True,
    gaussian=0,
    final_weight=1.0,
    total_epochs=55,
    visual_batch_video=True)

#mean=[127.317, 127.317, 127.317],
#std=[50.863, 50.863, 50.863],
dataset_type = 'SAMMRawFrameWithAUDataset'

data_root = '/path/to/your/datasets/SAMM/SAMM_ORI_ALL_CLASS20200525'
data_root_val = '/path/to/your/datasets/SAMM/SAMM_ORI_ALL_CLASS20200525'
data_root_test = '/path/to/your/datasets/SAMM/SAMM_ORI_ALL_CLASS20200525'
ann_file_train = 'data/samm/samm_5class_emolabel_list_ori.txt'
ann_file_val = 'data/samm/samm_5class_emolabel_list_ori.txt'
ann_file_test = 'data/samm/samm_5class_emolabel_list_ori.txt'
au_ann_file_train = 'data/samm/samm_5class_AU_list.txt'
au_ann_file_val = 'data/samm/samm_5class_AU_list.txt'
au_ann_file_test = 'data/samm/samm_5class_AU_list.txt'


file_client_args = dict(io_backend='disk')
'''
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames', clip_len=num_frames, frame_interval=4, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    #dict(type='RandomResizedCrop'),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, .875, .75, .66),
        random_crop=False,
        num_fixed_crops=13,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(
        type='PytorchVideoWrapper',
        op='AugMix',
        magnitude=3),
    #dict(type='RandomErasing', erase_prob=0.25, mode='rand'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
'''
train_pipeline = [
    dict(type='PrepareCASME3Info'),
    dict(type='UniformSample', clip_len=16, test_mode=False),
    dict(type='RawFrameDecode2'),
    #dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PytorchVideoWrapper', op='AugMix', magnitude=1),
    #dict(type='Flip', flip_ratio=0.5),
    #dict(type='ColorJitter'),
    #dict(type='RandomErasing'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='PrepareCASME3Info'),
    dict(type='UniformSample', num_clips=1, clip_len=16, test_mode=True),
    dict(type='RawFrameDecode2'),
    dict(type='Resize', scale=(-1, 224)),
    #dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='PrepareCASME3Info'),
    dict(type='UniformSample', num_clips=1, clip_len=16, test_mode=True),
    dict(type='RawFrameDecode2'),
    dict(type='Resize', scale=(-1, 224)),
    #dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        emo_ann_file=ann_file_train,
        au_ann_file=au_ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        emo_ann_file=ann_file_val,
        au_ann_file=au_ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        emo_ann_file=ann_file_test,
        au_ann_file=au_ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))
#"anger", "contempt", "happiness", "other", "surprise"
label_map = {"0": "anger", "1": "contempt", "2": "happiness", "3": "other", "4": "surprise"}

val_evaluator = dict(type='MEMetric', label_map=label_map)
test_evaluator = dict(type='MEMetric', save_results=True, label_map=label_map)

train_cfg = dict(
    type='WithEpochBasedTrainLoop', max_epochs=55, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
'''
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=5e-5, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.2),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=20, norm_type=2))
'''
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=5e-5, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=20, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min_ratio=0.1,
        T_max=50,
        by_epoch=True,
        begin=5,
        end=55,
        convert_to_iter_based=True)
]

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=256)

default_scope = 'mmaction'

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, save_best='acc/UF1', max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
load_from = None
resume = False
