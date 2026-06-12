_base_ = [
    '../_base_/models/tinycd.py',
    '../common/standard_256x256_40k_levircd.py']

crop_size = (256, 256)
model = dict(
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
