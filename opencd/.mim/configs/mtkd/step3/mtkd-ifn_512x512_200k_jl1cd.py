_base_ = [
    '../../_base_/models/mtkd/mtkd-ifn.py',
    '../../common/standard_512x512_200k_jl1cd.py']

checkpoint_student = None
checkpoint_teacher_l = None
checkpoint_teacher_m = None
checkpoint_teacher_s = None

model = dict(

    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_student),

    init_cfg_t_l = dict(type='Pretrained', checkpoint=checkpoint_teacher_l),

    init_cfg_t_m = dict(type='Pretrained', checkpoint=checkpoint_teacher_m),

    init_cfg_t_s = dict(type='Pretrained', checkpoint=checkpoint_teacher_s)
)
