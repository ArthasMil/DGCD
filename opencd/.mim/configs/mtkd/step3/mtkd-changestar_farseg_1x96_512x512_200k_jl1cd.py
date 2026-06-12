_base_ = [
    '../../_base_/models/mtkd/mtkd-changestar_farseg_1x96_r18.py',
    '../../common/standard_512x512_200k_jl1cd.py']

checkpoint_student = None
checkpoint_teacher_l = None
checkpoint_teacher_m = None
checkpoint_teacher_s = None

train_dataloader = dict(batch_size=16, num_workers=8)

model = dict(

    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_student),

    init_cfg_t_l = dict(type='Pretrained', checkpoint=checkpoint_teacher_l),

    init_cfg_t_m = dict(type='Pretrained', checkpoint=checkpoint_teacher_m),

    init_cfg_t_s = dict(type='Pretrained', checkpoint=checkpoint_teacher_s),
)

optimizer=dict(
    type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
