
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()

lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=500)
checkpoint_config = dict(by_epoch=True, interval=1)
evaluation = dict(by_epoch=True, interval=1, metric='mIoU', pre_eval=True)
