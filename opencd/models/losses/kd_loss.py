# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn

from opencd.registry import MODELS

def kd_loss(
    pred_s,
    pred_t,
    eps=1e-4,
    ignore_index=255,
    **kwargs):
    assert pred_s.size() == pred_t.size()
    mask = (pred_t != ignore_index).float()
    n_pixel = mask.sum() + eps
    loss = torch.sum(torch.pow(pred_s-pred_t, 2)) / n_pixel
    return loss

@MODELS.register_module()
class DistillLoss(nn.Module):

    def __init__(self,
                 temperature=1.0,
                 loss_weight=1.0,
                 loss_name='distill_loss',
                 **kwargs):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                student_pred,
                teacher_pred,
                **kwargs):

        student_logits = nn.functional.log_softmax(student_pred / self.temperature, dim=1)
        teacher_logits = nn.functional.softmax(teacher_pred / self.temperature, dim=1)

        distill_loss = nn.functional.kl_div(
            student_logits, teacher_logits, reduction='batchmean') * (self.temperature ** 2)

        num_elements = student_logits.numel()

        loss = self.loss_weight * distill_loss
        return loss

    @property
    def loss_name(self):
        return self._loss_name

@MODELS.register_module()

class DistillLossWithPixel(nn.Module):

    def __init__(self, temperature=1.0, loss_weight=1.0, pixel_weight=1.0, loss_name='distill_loss', **kwargs):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.pixel_weight = pixel_weight
        self._loss_name = loss_name

    def forward(self, student_pred, teacher_pred, **kwargs):

        student_logits = nn.functional.log_softmax(student_pred / self.temperature, dim=1)
        teacher_logits = nn.functional.softmax(teacher_pred / self.temperature, dim=1)
        class_distill_loss = nn.functional.kl_div(
            student_logits, teacher_logits, reduction='batchmean') * (self.temperature ** 2)

        pixel_distill_loss = nn.functional.mse_loss(student_pred, teacher_pred)

        total_loss = (self.loss_weight * class_distill_loss +
                      self.pixel_weight * pixel_distill_loss)
        return total_loss

    @property
    def loss_name(self):
        return self._loss_name

@MODELS.register_module()

class DistillLossWithPixel_ChangeStar(nn.Module):

    def __init__(self, temperature=1.0, loss_weight=1.0, pixel_weight=1.0, loss_name='distill_loss', **kwargs):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.pixel_weight = pixel_weight
        self._loss_name = loss_name

    def forward(self, student_pred, teacher_pred, **kwargs):
        student_pred = torch.cat([student_pred[0], student_pred[1]], dim=0)
        teacher_pred = torch.cat([teacher_pred[0], teacher_pred[1]], dim=0)

        student_logits = nn.functional.log_softmax(student_pred / self.temperature, dim=1)
        teacher_logits = nn.functional.softmax(teacher_pred / self.temperature, dim=1)
        class_distill_loss = nn.functional.kl_div(
            student_logits, teacher_logits, reduction='batchmean') * (self.temperature ** 2)

        pixel_distill_loss = nn.functional.mse_loss(student_pred, teacher_pred)

        total_loss = (self.loss_weight * class_distill_loss +
                      self.pixel_weight * pixel_distill_loss)
        return total_loss

    @property
    def loss_name(self):
        return self._loss_name
