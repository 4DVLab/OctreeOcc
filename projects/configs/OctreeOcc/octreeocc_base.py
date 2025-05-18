_base_ = [
    '/public/home/luyh2/PanoOcc/projects/configs/datasets/custom_nus-3d.py',
    '/public/home/luyh2/PanoOcc/projects/configs/_base_/default_runtime.py'
]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point cloud range accordingly
# Occ3d-nuScenes in ego coordinate
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

ida_aug_conf = {
        "resize_lim": (1.0, 1.0),
        "final_dim": (900, 1600),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
}

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
# Voxel Queries Size
bev_h_ = [50,100,200]
bev_w_ = [50,100,200]
bev_z_ = [4,8,16]
# Temporal 
queue_length = 16 # each sequence contains `queue_length` frames.
key_frame = 3
time_interval = 4

model = dict(
    type='OctreeOcc_pred',
    use_grid_mask=True,
    video_test_mode=True,
    time_interval = time_interval,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp = True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='OctreeOccHead',
        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
        bev_z=bev_z_,
        num_classes=18,
        in_channels=_dim_,
        use_init_pred=False,
        use_octree_embed=True,
        sync_cls_avg_factor=True,
        with_box_refine=False,
        as_two_stage=False,
        use_mask=True,
        loss_occ= dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=10.0),
        loss_det_occ= dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_octree_pred = dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_octree_pred = dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0),
        loss_occupancy_aux = dict(
            type = 'Lovasz3DLoss',
            ignore = 17,
            loss_weight=1.0),
        transformer=dict(
            type='OctreeOccTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            num_feature_levels = _num_levels_,
            cam_encoder=dict(
                type='OctreeOccupancyEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=bev_z_,
                return_intermediate=False,
                ego = True,
                transformerlayers=dict(
                    type='OctreeOccupancyLayer',
                    attn_cfgs=[
                        dict(
                            type='OctreeSelfAttention3D',
                            embed_dims=_dim_,
                            num_points=4, 
                            num_levels=1),
                        dict(
                            type='OccSpatialAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points= 8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
            seg_decoder = dict(
                type='OctreeDecoder',
                embed_dims = _dim_,
                out_dim = _dim_ // 8,
                num_classes = 18,
                pc_range=point_cloud_range,
                output_size=[200,200,16],
                bev_h=bev_h_,
                bev_w=bev_w_,
                bev_z=bev_z_,)),
        octree_positional_encoding=dict(
            type='OctreePositionEmbedding',
            num_pos_feats=_dim_,
            ),

    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range)))))

dataset_type = 'NuScenesOcc'
data_root = 'data/occ3d-nus/'
file_client_args = dict(backend='disk')
occ_gt_data_root='data/occ3d-nus'

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile_octree',data_root=occ_gt_data_root),
    dict(type='LoadSegPriorFromFile',data_root=occ_gt_data_root),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[1.0]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=[ 'img','voxel_semantics','mask_lidar','mask_camera','gt_bboxes_3d', 'gt_labels_3d',
                                        'subdividable_different_level_l1', 'subdividable_different_level_l2',
                                        'seg_structure_l1', 'seg_structure_l2'] )
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='LoadOccGTFromFile_octree',data_root=occ_gt_data_root),
    dict(type='LoadSegPriorFromFile',data_root=occ_gt_data_root),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[1.0]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img', 'subdividable_different_level_l1', 'subdividable_different_level_l2', 'seg_structure_l1', 'seg_structure_l2'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'occ_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        key_frame = key_frame,
        time_interval = time_interval,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'occ_infos_temporal_val.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'occ_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 240
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=120)