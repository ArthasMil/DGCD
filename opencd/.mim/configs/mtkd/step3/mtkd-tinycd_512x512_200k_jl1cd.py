_base_ = [
    '../../_base_/models/mtkd/mtkd-tinycd.py',
    '../../common/standard_512x512_200k_jl1cd.py']

checkpoint_student = None
checkpoint_teacher_l = None
checkpoint_teacher_m = None
checkpoint_teacher_s = None

model = dict(

    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_student),

    init_cfg_t_l = dict(type='Pretrained', checkpoint=checkpoint_teacher_l),

    init_cfg_t_m = dict(type='Pretrained', checkpoint=checkpoint_teacher_m),

    init_cfg_t_s = dict(type='Pretrained', checkpoint=checkpoint_teacher_s),

    decode_head=dict(num_classes=2, out_channels=1),
)

optimizer = dict(
    type='AdamW',
    lr=0.00356799066427741,
    betas=(0.9, 0.999),
    weight_decay=0.009449677083344786)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer)
